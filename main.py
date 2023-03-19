
import os
import torch
import transformers
from transformers import GPT2Config
from utils import process_data,read_data,logger,init_gpu_params
from models import Re_Distiller
# train_generator_MLE
import student 
import argparse
import json
import numpy as np
def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def sanity_checks(args):
    """
    A bunch of args sanity checks to perform even starting...
    """
    assert (args.mlm and args.alpha_mlm > 0.0) or (not args.mlm and args.alpha_mlm == 0.0)
    assert (args.alpha_mlm > 0.0 and args.alpha_clm == 0.0) or (args.alpha_mlm == 0.0 and args.alpha_clm > 0.0)
    if args.mlm:
        assert os.path.isfile(args.token_counts)
        assert (args.student_type in ["roberta", "distilbert", "gpt2"]) and (args.teacher_type in ["roberta", "bert", "gpt2"])
    else:
        assert (args.student_type in ["gpt2"]) and (args.teacher_type in ["gpt2"])

    assert args.teacher_type == args.student_type or (
        args.student_type == "distilbert" and args.teacher_type == "bert"
    )
    assert os.path.isfile(args.student_config)
    if args.student_pretrained_weights is not None:
        assert os.path.isfile(args.student_pretrained_weights)

    if args.freeze_token_type_embds:
        assert args.student_type in ["roberta"]

    assert args.alpha_ce >= 0.0
    assert args.alpha_mlm >= 0.0
    assert args.alpha_clm >= 0.0
    assert args.alpha_mse >= 0.0
    assert args.alpha_cos >= 0.0
    assert args.alpha_ce + args.alpha_mlm + args.alpha_clm + args.alpha_mse + args.alpha_cos > 0.0


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--force", action="store_true", help="Overwrite dump_path if it already exists.")

    parser.add_argument(
        "--dump_path", type=str, required=True, help="The output directory (log, checkpoints, parameters, etc.)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="The binarized file (tokenized + tokens_to_ids) and grouped by sequence.",
    )

    parser.add_argument(
        "--student_type",
        type=str,
        choices=["distilbert", "roberta", "gpt2"],
        required=True,
        help="The student type (DistilBERT, RoBERTa).",
    )
    parser.add_argument("--student_config", type=str, required=True, help="Path to the student configuration.")
    parser.add_argument(
        "--student_pretrained_weights", default=None, type=str, help="Load student initialization checkpoint."
    )

    parser.add_argument(
        "--teacher_type", choices=["bert", "roberta", "gpt2"], required=True, help="Teacher type (BERT, RoBERTa)."
    )
    parser.add_argument("--teacher_name", type=str, required=True, help="The teacher model.")

    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for the softmax temperature.")
    parser.add_argument(
        "--alpha_ce", default=0, type=float, help="Linear weight for the distillation loss. Must be >=0."
    )
    parser.add_argument(
        "--alpha_mlm",
        default=0.0,
        type=float,
        help="Linear weight for the MLM loss. Must be >=0. Should be used in coonjunction with `mlm` flag.",
    )
    parser.add_argument("--alpha_clm", default=1, type=float, help="Linear weight for the CLM loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.0, type=float, help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument(
        "--alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0."
    )

    parser.add_argument(
        "--mlm", action="store_true", help="The LM step: MLM or CLM. If `mlm` is True, the MLM is used over CLM."
    )
    parser.add_argument(
        "--mlm_mask_prop",
        default=0.15,
        type=float,
        help="Proportion of tokens for which we need to make a prediction.",
    )
    parser.add_argument("--word_mask", default=0.8, type=float, help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float, help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float, help="Proportion of tokens to randomly replace.")
    parser.add_argument(
        "--mlm_smoothing",
        default=0.7,
        type=float,
        help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).",
    )
    parser.add_argument("--token_counts", type=str, help="The token counts in the data_file for MLM.")

    parser.add_argument(
        "--restrict_ce_to_mask",
        action="store_true",
        help="If true, compute the distilation loss only the [MLM] prediction distribution.",
    )
    parser.add_argument(
        "--freeze_pos_embs",
        action="store_true",
        help="Freeze positional embeddings during distillation. For student_type in ['roberta', 'gpt2'] only.",
    )
    parser.add_argument(
        "--freeze_token_type_embds",
        action="store_true",
        help="Freeze token type embeddings during distillation if existent. For student_type in ['roberta'] only.",
    )

    parser.add_argument("--n_epoch", type=int, default=2, help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (for each process).")
    parser.add_argument(
        "--group_by_size",
        action="store_true",
        help="If true, group sequences that have similar length into the same batch. Default is true.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument("--warmup_prop", default=0.05, type=float, help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.25, type=float, help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval.")
    args = parser.parse_args()
    init_gpu_params(args)
    sanity_checks(args)
    set_seed(args)
    if args.is_master:
        if os.path.exists(args.dump_path):
            if not args.force:
                raise ValueError(
                    f"Serialization dir {args.dump_path} already exists, but you have not precised wheter to overwrite it"
                    "Use `--force` if you want to overwrite it"
                )
            else:
                print("test")
                #shutil.rmtree(args.dump_path)

        if not os.path.exists(args.dump_path):
            os.makedirs(args.dump_path)
        logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

        # SAVE PARAMS #
        logger.info(f"Param: {args}")
        with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
            json.dump(vars(args), f, indent=4)





    # load the model for student, teacher, discriminator
    student_config_class = GPT2Config.from_json_file('Gan-Distill/config2.json')
    teacher_config_class = GPT2Config.from_json_file('Gan-Distill/config.json')
    dis_config_class = GPT2Config.from_json_file('Gan-Distill/config3.json')
    student_config_class.output_hidden_states = True
    teacher_config_class.output_hidden_states = True
    dis_config_class.output_hidden_states = True

    student_model = student.Student(student_config_class)
    teacher_model = student.Student(teacher_config_class)
    
    # discriminator_model = student.Discriminator(dis_config_class)
    discriminator_model = student.Discriminator(256,64,256,40,True,0.2)
    teacher_state_dict_path = torch.load(open(os.path.join("Gan-Distill/unigram", 'model.pt'), 'rb'), map_location=lambda s, l: s)
    # load the teacher model from pre-trained path
    teacher_model.load_state_dict(teacher_state_dict_path)
    student_state_dict_path = torch.load(open(os.path.join("Gan-Distill/temp", 'model.pt'), 'rb'), map_location=lambda s, l: s)
    student_model.load_state_dict(student_state_dict_path)

    # load to GPU

    if args.n_gpu > 0:
        student_model.to(f"cuda:{args.local_rank}")
    logger.info("Student loaded.")


    if args.n_gpu > 0:
        teacher_model.to(f"cuda:{args.local_rank}")
    logger.info("Teacher loaded.")

 
    if args.n_gpu > 0:
        discriminator_model.to(f"cuda:{args.local_rank}")
    logger.info("dis_model loaded.")

    #oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    data_txt = read_data()
    token_probs = None
    #oracle_samples, vocabulary, reverse_vocab, sentence_lengths = read_sampleFile(num=1402)


    torch.cuda.empty_cache()
    re_distiller = Re_Distiller(
    params=args, dataset=data_txt, token_probs=token_probs, student=student_model, teacher=teacher_model, discriminator=discriminator_model
    )
    # for i in range(10):
    #     re_distiller.train_generator_MLE()
      
    for i in range(20):
        # logger.info("Let's go train generator_MLE().")
        # re_distiller.train_generator_MLE()
        logger.info("Let's go train discriminator().")
        re_distiller.train_discriminator()
        logger.info("Let's go train generator().")
        
        re_distiller.train_generator()
    logger.info("Let's save the generator().")
    re_distiller.train_generator(save=True)
        # logger.info("Let's go train generator_MLE().")
    #     # for i in range(2):
    #     #     re_distiller.train_generator_MLE()
    # logger.info("Let's go get some drinks.")
    # re_distiller.train_generator_MLE()
    # for i in range(10):
        
    # re_distiller.train_generator_PG()
        
    #     re_distiller.train_discriminator()


    # MLE_TRAIN_EPOCHS = 10
    # train_generator_MLE(student_model, stu_optimizer, teacher_model, oracle_samples, MLE_TRAIN_EPOCHS)
    #VOCAB_SIZE = len(vocabulary)
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()