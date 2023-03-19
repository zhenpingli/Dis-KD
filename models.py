from __future__ import print_function
import math
import os
import time
from tqdm import tqdm
from math import ceil
import numpy as np
import sys
import pdb


import torch
import torch.nn.functional as F
from torch import nn



from utils import  logger
import student 

import math
import os
import time

import psutil
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler,SequentialSampler
import transformers

from lm_seqs_dataset import LmSeqsDataset
from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups

from tokenization import Tokenizer
from utils import logger, parse_sample_options, collate_fn, prepare_discriminator_data, prepare_generator_batch, prepare_generator_data

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

START_LETTER = 0
CUDA = True

def get_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))


class Re_Distiller:
    def __init__(
        self, params: dict, dataset: LmSeqsDataset, token_probs: torch.tensor, student: nn.Module, teacher: nn.Module, discriminator: nn.Module
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher
        self.model_params = get_params(student)
        self.discriminator = discriminator
        self.dis_model_params = get_params(discriminator)

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
            #sampler = SequentialSampler(dataset)
        
     

        # if params.group_by_size:
        #     groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
        #     sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        # else:
        #     sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        spm_path = os.path.join('/home/Gan-Distill/spm', "unigram", "spm.model")
        tokenizer = Tokenizer(spm_path)
        #sample = parse_sample_options([-1,0.2])
        sample = parse_sample_options(None)
        #self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=lambda x: collate_fn(x, tokenizer, sample, 40))
        self.dataloader = DataLoader(dataset=dataset, batch_size = params.batch_size,drop_last=True,shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer, sample, 40))

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        # self.alpha_ce  = 0.0
        # self.alpha_mlm = params.alpha_mlm
        # self.alpha_clm = 1
        # self.alpha_mse = params.alpha_mse
        # self.alpha_cos = 0.0

        self.mlm = params.mlm
        if self.mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
            self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs
            self.token_probs = token_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else token_probs
            if self.fp16:
                self.pred_probs = self.pred_probs.half()
                self.token_probs = self.token_probs.half()
        else:
            logger.info("Using CLM loss for LM step.")

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
       
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        #no_decay = ["bias", "LayerNorm.weight"]
        no_decay = []
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        # self.optimizer = AdamW(
        #     optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        # )
        self.optimizer = transformers.AdamW(optimizer_grouped_parameters)
        self.dis_optimizer = transformers.AdamW(self.discriminator.parameters())
        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        # )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )

        logger.info("--- Initializing Tensorboard")
        self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
        self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
        self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)



    def prepare_batch_clm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lable_ids, lengths, mask_attn = batch
        #token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        #assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, lable_ids, mask_attn

    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor, mask_attn: torch.tensor, return_mle=False):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        if self.mlm:
            student_outputs = self.student(
                input_ids=input_ids, attention_mask=attention_mask
            )  # (bs, seq_length, voc_size)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )  # (bs, seq_length, voc_size)
        else:
            student_outputs = self.student(input_ids=input_ids, attention_mask=None)  # (bs, seq_length, voc_size)
            with torch.no_grad():
                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=None)  # (bs, seq_length, voc_size)
        
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        assert s_logits.size() == t_logits.size()

        # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
        if self.params.restrict_ce_to_mask:
            mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        else:
            mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        if self.alpha_mlm > 0.0:
            loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
            loss += self.alpha_mlm * loss_mlm
        if self.alpha_clm > 0.0:
            #shift_logits = s_logits[..., :-1, :].contiguous()
            shift_logits = s_logits.contiguous()
            #shift_labels = lm_labels[..., 1:].contiguous()
            shift_labels = lm_labels.contiguous()
            raw_loss = F.cross_entropy(shift_logits.view(-1, 256), shift_labels.view(-1), reduction='none')
            bsz = input_ids.size(0)
            raw_loss = raw_loss.view(bsz, -1)
            temp = raw_loss * mask_attn.float()
            temp2 = temp.sum(1)
            loss_clm = temp2.mean()
            #loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm
        if return_mle == True:
            return loss
        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            nn.utils.clip_grad_norm_(self.model_params, 0.25)
            nn.utils.clip_grad_norm_(self.dis_model_params, 0.25)
            loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
    

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(
            tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter
        )
        if self.alpha_mlm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mlm", scalar_value=self.last_loss_mlm, global_step=self.n_total_iter
            )
        if self.alpha_clm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_clm", scalar_value=self.last_loss_clm, global_step=self.n_total_iter
            )
        if self.alpha_mse > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter
            )
        if self.alpha_cos > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_cos", scalar_value=self.last_loss_cos, global_step=self.n_total_iter
            )
        # self.tensorboard.add_scalar(
        #     tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        # )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        
        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if hasattr(self.student, "module"):
            print("test")
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))


    #train_generator_MLE(student_model, stu_optimizer, teacher_model, oracle_samples, MLE_TRAIN_EPOCHS)
    def train_generator_MLE(self):
        """
        Max Likelihood Pretraining for the generator
        """
    
        logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()
    


        #for _ in range(self.params.n_epoch):
        for _ in range(self.params.n_epoch):
            
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
    
            
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                token_ids, attn_mask, lm_labels, mask_attn = self.prepare_batch_clm(batch=batch)
                self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels, mask_attn=mask_attn)

                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_bar.close()


        
            
            logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        
        # logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        # self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        # logger.info("Training is finished")


    def train_discriminator(self):

        logger.info("Starting training")
        self.last_log = time.time()
        self.student.eval()
        self.teacher.eval()
        self.discriminator.train()
        #for _ in range(self.params.n_epoch):
        for _ in range(1):
            logger.info(f"--- Starting epoch ---")
            cont = 0
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                cont = cont +1 
                token_ids, attn_mask, lm_labels, mask_attn = self.prepare_batch_clm(batch=batch)

                #self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels, mask_attn=mask_attn)
                self.dis_optimizer.zero_grad()
                student_outputs = self.student(input_ids=token_ids, attention_mask=attn_mask)  # (bs, seq_length, voc_size)
                s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]


                
                teacher_outputs = self.teacher(input_ids=token_ids, attention_mask=attn_mask)  # (bs, seq_length, voc_size)
                t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]

                dis_inp, dis_target = prepare_discriminator_data(t_logits.cpu(), s_logits.cpu(), gpu=True)
                epochs = 1
                total_acc = 0
                total_loss = 0
             
                    #print('d-step %d epoch %d : ' % (epoch + 1, epoch + 1), end='')
          
                
                #inp, target = dis_inp, dis_target
                
                out = self.discriminator.batchClassify(dis_inp)
                loss = self.discriminator.batchBCELoss(out, dis_target)
                loss.backward()
                self.dis_optimizer.step()
                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(dis_target>0.5)).data.item()

                #if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        #BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
      
              

                total_loss /= float(16)
                total_acc /= float(dis_inp.shape[0]*epochs)
                if cont%100 == 1:
                    print(' average_loss = %.4f, train_acc = %.4f, ' % (total_loss, total_acc ))
                if cont>128:
                    break 

    def train_generator(self,save=False):

        logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()
        
        #self.discriminator.eval()
        #for _ in range(self.params.n_epoch):
        for _ in range(1):
            logger.info(f"--- Starting epoch ---")
            
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])

            total_acc = 0
            total_loss = 0
            mse_loss = 0
            pg_loss = 0 
            dis_loss = 0
            n = 0
            for batch in iter_bar:
                n = n+1
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                token_ids, attn_mask, lm_labels, mask_attn = self.prepare_batch_clm(batch=batch)

                #self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels, mask_attn=mask_attn)
                #with torch.no_grad():
                self.optimizer.zero_grad()
                student_outputs = self.student(input_ids=token_ids, attention_mask=attn_mask)  # (bs, seq_length, voc_size)
             
                s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
               
                cont_nan = torch.isnan(s_logits).int().sum()
                if cont_nan > 10:
                    print("out put is nan")
                    continue
                


                with torch.no_grad():
                    teacher_outputs = self.teacher(input_ids=token_ids, attention_mask=attn_mask)  # (bs, seq_length, voc_size)
                t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]

                dis_inp, dis_target = prepare_generator_data(s_logits, s_logits, gpu=True)
      
                #print('d-step %d epoch %d : ' % (epoch + 1, epoch + 1), end='')
                #sys.stdout.flush()
                
                #inp, target = dis_inp, dis_target
                
            
    
                
                # discriminator loss
                #with torch.no_grad():
                out = self.discriminator.batchClassify(dis_inp)
                #out = out.detach()
             
          
           
                loss_dis = self.student.batchBCELoss(out, dis_target)
      
      
                dis_loss += loss_dis.data.item()
                



                # MSE loss
                loss_mse = self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels, mask_attn=mask_attn, return_mle=True)
                mse_loss += loss_mse

                # policy gradient loss

                # argmax

                # pl_softmax = F.log_softmax(s_logits,dim=2)
                # pl_prob, pl_out = torch.max(pl_softmax, dim=2)
                
                # argmax
               

                # 1-argmax
                pl_softmax = F.softmax(s_logits,dim=2)
                pl_prob, pl_out = torch.max(pl_softmax, dim=2)
               
                pl_prob = 1-pl_prob
                #pl_prob = torch.div(pl_prob, 4.837)
                pl_prob = torch.log(pl_prob)
                # 1-argmax

                #inp, target = prepare_generator_batch(pl_out, start_letter=START_LETTER, gpu=CUDA)
                #with torch.no_grad():
                rewards = self.discriminator.batchClassify(s_logits)
                # rewards = self.discriminator.batchseqClassify(target)
            
                #loss_pg = self.student.batchPGLoss(pl_out, pl_out, rewards)
               
                loss_pg = self.student.batchPGLoss(pl_prob, pl_out, rewards)
              
                pg_loss += loss_pg.data.item()



                # total loss
                # 0.1 0.75 0.15
                # 0.1 0.9
                gen_loss = 0.6*loss_mse + 0.35*loss_pg
                # if loss_pg.data.item()>5*loss_mse.data.item():
                #     gen_loss = 0.1*loss_dis + 0.75*loss_mse + 0.02*loss_pg
                # nn.utils.clip_grad_norm_(self.model_params, 0.25)
                # nn.utils.clip_grad_norm_(self.dis_model_params, 0.25)
                #loss.backward()
                
                gen_loss.backward()
                self.optimizer.step()
                total_loss += gen_loss.data.item()
                
                total_acc += torch.sum((out<0.5)==(dis_target<0.5)).data.item()
                
                total_acc /= float(dis_inp.shape[0]*n)
                print(' average_loss = %.4f, mse_loss =  %.4f,  pg_loss =  %.4f, dis_loss =  %.4f, train_acc = %.4f, ' % (total_loss/n, mse_loss/n, pg_loss/n, dis_loss/n, total_acc)) 
                #if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        #BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
             
                if n>256:
                    if save == True:
                        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
                        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
                    break

                #total_acc /= float(dis_inp.shape[0])
            print(' average_loss = %.4f, mse_loss =  %.4f,  pg_loss =  %.4f, dis_loss =  %.4f, train_acc = %.4f, ' % (total_loss/16, mse_loss/16, pg_loss/16, dis_loss/16, total_acc)) 
            #print(' average_loss = %.4f, mse_loss =  %.4f,  dis_loss =  %.4f, train_acc = %.4f, ' % (total_loss, mse_loss, dis_loss, total_acc)) 
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")


    def train_generator_PG(self):

        logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()
        self.discriminator.train()

        for _ in range(1):
            
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                token_ids, attn_mask, lm_labels, mask_attn = self.prepare_batch_clm(batch=batch)

                student_outputs = self.student(input_ids=token_ids, attention_mask=attn_mask)  # (bs, seq_length, voc_size)
                s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
                out = nn.LogSoftmax(s_logits, dim=2)
                
                out = torch.exp(out).argmax(dim=2)

                inp, target = prepare_generator_batch(out, start_letter=START_LETTER, gpu=CUDA)

                rewards = self.discriminator.batchseqClassify(target)
                self.optimizer.zero_grad()
                pg_loss = self.student.batchPGLoss(inp, target, rewards)
                
                pg_loss.backward()
                self.optimizer.step()
                print(' pg_loss = %.4f' % pg_loss)
        # logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        # self.save_checkpoint(checkpoint_name="pg_pytorch_model.bin")
        # logger.info("Training is finished")

                





