import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.autograd as autograd
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN

from transformers import GPT2Config
from transformers.models.gpt2 import GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2 import GPT2PreTrainedModel
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import torch.nn.functional as F



class Student(GPT2PreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

 
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True


    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


    
    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        # inp = inp.permute(1, 0)          # seq_len x batch_size
        # target = target.permute(1, 0)    # seq_len x batch_size

        # loss = 0
        # for i in range(seq_len):
        #     # out = self.forward(inp[i])
        #     # out = out["logits"]
        #     # out = F.log_softmax(out, dim=1)
        #     # TODO: should h be detached from graph (.detach())?
        #     out = inp
        #     for j in range(batch_size):
        #         #loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q
        #         loss += -out[i][j]*reward[j]

        loss = -torch.mul(torch.sum(inp,dim=-1),reward)
        loss = torch.sum(loss,dim=-1)

        return loss/batch_size


    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        return loss_fn(inp, target)
        
    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor, mask_attn: torch.tensor):

        student_outputs = self.student(input_ids=input_ids, attention_mask=None)  # (bs, seq_length, voc_size)

# class Discriminator(GPT2PreTrainedModel):

#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = GPT2Model(config)
#         self.dropout_linear = nn.Dropout(p=0.2)
#         #self.hidden2out = nn.Linear(config.n_embd, 1)
#         self.gru = nn.GRU(256, 256, num_layers=2, bidirectional=True, dropout=0.2)
#         self.gru2hidden = nn.Linear(4*config.n_embd, 256)
#         #self.trans = nn.Linear(256,512)
#         self.hidden2out = nn.Linear(config.n_embd, 1)

#         self.init_weights()

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#     def init_hidden(self, batch_size):
#         h = autograd.Variable(torch.zeros(2*2*1, batch_size, 256))

        
#         return h.cuda()
        

#     def forward(
#         self,
#         input_ids=None,
#         past_key_values=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
#             ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
#             ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         h = self.init_hidden(hidden_states.size()[0])
#         hidden_states = hidden_states.permute(1, 0, 2)

#         _, hidden = self.gru(hidden_states, h) 
#         hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
#         out = self.gru2hidden(hidden.view(-1, 4*256))  # batch_size x 4*hidden_dim

     

#         # input dim                                                # batch_size x seq_len

#         out = torch.tanh(out)
#         out = self.dropout_linear(out)
#         out = self.hidden2out(out)                                 # batch_size x 1
        
#         #out = torch.sigmoid(out)
#         out = torch.sigmoid(out) 
        
#         return out


#     def batchClassify(self, inp):
#         """
#         Classifies a batch of sequences.

#         Inputs: inp
#             - inp: batch_size x seq_len

#         Returns: out
#             - out: batch_size ([0,1] score)
#         """

      
#         out = self.forward(inputs_embeds=inp)
#         #out = out[:,-1,:]
#         return out.view(-1)
    
#     def batchseqClassify(self, inp):
#         """
#         Classifies a batch of sequences.

#         Inputs: inp
#             - inp: batch_size x seq_len

#         Returns: out
#             - out: batch_size ([0,1] score)
#         """

      
#         out = self.forward(inp)
#         #out = out[:,-1,:]
#         return out.view(-1)    

#     def batchBCELoss(self, inp, target):
#         """
#         Returns Binary Cross Entropy Loss for discriminator.

#          Inputs: inp, target
#             - inp: batch_size x seq_len
#             - target: batch_size (binary 1/0)
#         """

#         loss_fn = nn.BCELoss()
#         return loss_fn(inp, target)

#     def batchMSELoss(self, inp, target):
#         """
#         Returns Binary Cross Entropy Loss for discriminator.

#          Inputs: inp, target
#             - inp: batch_size x seq_len
#             - target: batch_size (binary 1/0)
#         """

#         loss_fn = nn.MSELoss()
#         return loss_fn(inp, target)    


class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        #self.activations = torch.nn.Sigmoid()
        self.activations = torch.nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input=None, hidden=None,inputs_embeds=None):
        # input dim                                                # batch_size x seq_len
        if inputs_embeds == None:
            emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
        else:
            emb = inputs_embeds
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = self.activations(out)
        #out = nn.LeakyReLU(out)
        return out

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

      
        # out = self.forward(inputs_embeds=inp)
        # #out = out[:,-1,:]
        # return out.view(-1)
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inputs_embeds = inp, hidden =h)
        return out.view(-1)
    

    def batchseqClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

      
        out = self.forward(inp)
        #out = out[:,-1,:]
        return out.view(-1)    

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """
        loss_fn = nn.BCELoss()
        # loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(inp, target)
 

