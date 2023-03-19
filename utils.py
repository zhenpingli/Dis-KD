import logging
import os
import numpy as np
import socket

from lm_seqs_dataset import LmSeqsDataset
from tokenization import Tokenizer

from torch.utils.data import BatchSampler, DataLoader, RandomSampler
import torch
from torch.autograd import Variable

#logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# read the data from dump

def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    inp = torch.cat((pos_samples.cuda(), neg_samples.cuda()), 0).type(torch.FloatTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[:pos_samples.size()[0]] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_generator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """
    inp = torch.cat((pos_samples.cuda(), neg_samples.cuda()), 0).type(torch.FloatTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    #target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target



def read_data():
    queries = []
    with open("Gan-Distill/data/dump.txt") as f:
        for x in f:
            x = x.rstrip('\n')
            if len(x) >= 3:
                queries.append(x)
        logger.info(f"  Number of {x:>5s} data: {len(queries):8d}")
    return queries

# process the data to Pytorch Dataloader formate

def process_data(dataset: LmSeqsDataset):

    sampler = RandomSampler(dataset)
    spm_path = os.path.join('Gan-Distill/spm', "unigram", "spm.model")
    tokenizer = Tokenizer(spm_path)
    sample = parse_sample_options([-1,0.2])
    dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=lambda x: collate_fn(x, tokenizer, sample, 40))
    
    return dataloader


def parse_sample_options(x):
    opt = {}
    if x:
        if len(x) == 2:
            opt = {'l': int(x[0]), 'alpha': x[1]}
        elif len(x) == 1:
            opt = {'n': int(x[0])}
    return opt


def collate_fn(queries, tokenizer, sample, max_seq_len=None):
    token_id_seqs = [[1] + tokenizer(x, **sample) + [2] for x in queries]

    length = [len(x) - 1 for x in token_id_seqs]
    if max_seq_len is None or max_seq_len > max(length) + 1:
        max_seq_len = max(length) + 1

    padded = []
    mask = []
    for x in token_id_seqs:
        x = x[:max_seq_len]
        pad_length = max_seq_len - len(x)
        padded.append(x + [0] * pad_length)
        mask.append([1] * (len(x) - 1) + [0] * pad_length)
      

    padded = torch.tensor(padded)
    length = torch.tensor(length)
    mask_attn = torch.tensor(mask)
    a = padded[:,:-1]
    b = padded[:,1:]
    return a, b, length, mask_attn



def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert params.local_rank == -1

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)
    print(params.local_rank)
    print(params.multi_gpu)

    # initialize multi-GPU
    # if params.multi_gpu:
    #     logger.info("Initializing PyTorch distributed")
    #     torch.distributed.init_process_group(
    #         init_method="env://",
    #         backend="nccl",
    #     )


def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    # inp = torch.zeros(batch_size, seq_len)
    inp =samples
    target = torch.zeros(batch_size, seq_len)
    target[:,:seq_len-1] = inp[:,1:]

    # target = samples
    # inp[:, 0] = 1
    # inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target