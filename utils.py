"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
import time
import math
import torch

from model import LMConfig, LanguageModel
from transformers import  GPT2Config
from GPT2LMHeadModel import GPT2LMHeadModel
class TrainLogger(object):
    def __init__(self):
        self.init()

    def init(self):
        self.start = time.time()
        self.cnt_add = 0
        self.tot_loss = 0.
        self.cnt_query = 0
        self.cnt_token = 0

    def add(self, loss, n_query, n_token):
        self.cnt_add += 1
        self.tot_loss += loss * n_query
        self.cnt_query += n_query
        self.cnt_token += n_token

    def average(self):
        loss_query = self.tot_loss / self.cnt_query if self.cnt_query != 0 else 0.
        loss_token = self.tot_loss / self.cnt_token if self.cnt_token != 0 else 0.
        return loss_query, loss_token

    def elapsed_time(self):
        return time.time() - self.start

    def print_str(self, time_avg_=False):
        loss_query, loss_token = self.average()
        time_str = f"{self.elapsed_time() * 1000. / self.cnt_add:6.2f} ms/batch" if time_avg_ else \
                   f"{self.elapsed_time():6.2f} s"
        return f"{time_str} | loss_query {loss_query:6.2f} | token_ppl {math.exp(loss_token):6.2f}"


def get_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def get_model(model):
    return model.module if hasattr(model, 'module') else model


def model_save(path, model, optimizer=None):
    model_to_save = get_model(model)
    open(os.path.join(path, 'config.json'), 'w').write(str(model_to_save.config))
    torch.save(model_to_save.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
    if optimizer:
        torch.save(optimizer.state_dict(), open(os.path.join(path, 'optimizer.pt'), 'wb'))


def model_load(path, model=None, optimizer=None):
    #config = LMConfig(os.path.join(path, 'config.json'))
    config_gpt =GPT2Config.from_json_file('config.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #GPT2Config.from_json_file(output_config_file)

    if model is None:
        model_to_load = GPT2LMHeadModel(config_gpt).to(device)
        #model_to_load = LanguageModel(config)
    else:
        model_to_load = get_model(model)
        model_to_load.__init__(config_gpt)
    model_state_dict = torch.load(open(os.path.join(path, 'model.pt'), 'rb'), map_location=lambda s, l: s)
    model_to_load.load_state_dict(model_state_dict)
    if optimizer:
        optimizer_state_dict = torch.load(open(os.path.join(path, 'optimizer.pt'), 'rb'), map_location=lambda s, l: s)
        optimizer.load_state_dict(optimizer_state_dict)
    return model_to_load
