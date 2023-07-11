# -*- encoding: utf-8 -*-
# @Author: Nafis Faiyaz
# @Time: 2023/03/07 21:46:05
# @File: deberta_embedder.py

import torch
from torch import nn
from transformers import DebertaModel

class DebertaEmbedder(nn.Module):
    def __init__(self,input_size,pretrained_model_path):
        super(DebertaEmbedder,self).__init__()
        self.deberta=DebertaModel.from_pretrained(pretrained_model_path)
        
    def forward(self,input_seq):
        output=self.deberta(input_seq)[0]
        return output
    def token_resize(self,input_size):
        self.deberta.resize_token_embeddings(input_size)