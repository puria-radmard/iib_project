import os, torch
from torch import nn
from torch.nn.init import xavier_normal_, zeros_
import json
device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)
from util_functions.data import generate_data_dict_words, generate_coll_fn_simclr_utt
# from util_functions.bert import get_all_sentences, pretrain_bert, generate_word_embedding_cache


def load_state_dict(model, weights_path):
    if weights_path != None:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

def get_config_from_file(file_path):
    with open(file_path, 'r') as jfile:
        kw = json.load(jfile)
    return kw

def length_aware_loss(crit_output, lengths):
    loss_mask = torch.ones_like(crit_output)
    for j, lm in enumerate(loss_mask):
        lm[lengths[j]:] = 0
    crit_output *= loss_mask
    return crit_output, loss_mask

def batch_trace(x):
    torch.diagonal(x, dim1=-2, dim2=-1).sum(-1) 

def batch_outer(z_i, z):
    return torch.einsum('bi,bj->bij', (z_i-z, z_i,z))
