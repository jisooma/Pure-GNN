import torch

import math
import torch
from torch import nn
from torch_geometric.nn import Linear

class Attention(nn.Module):
    def __init__(self,**kwargs):
        super(Attention,self).__init__()

    def forward(self,queries,keys,values):
        raise NotImplementedError

class At(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0.5,**kwargs):
        super(Attention,self).__init__()
        self.score_func = nn.Sequential(
            nn.Linear(key_size,num_hiddens),
            nn.Tanh(),
            nn.Linear(num_hiddens,1,bias=False)
        )
    def forward(self,queries,keys,values):

        w = self.score_func(values)
        alpha = torch.softmax(w,dim=1)
        aggr_max = torch.argmax(alpha,dim=1)
        res = (alpha*values).sum(1)
        return res,aggr_max,alpha


class AdditiveAttention(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0,**kwargs):
        super(AdditiveAttention,self).__init__(**kwargs)

        self.W_k = Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')
        self.W_q = Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')
        self.W_v = Linear(num_hiddens, 1, bias=False,weight_initializer='glorot')
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values):

        queries,keys = self.W_q(queries),self.W_k(keys)
        features = torch.tanh(queries+keys)
        scores = self.W_v(features)

        alpha = torch.softmax(scores,dim=1)
        alpha = self.dropout(alpha)

        aggr_max = torch.argmax(alpha, dim=1)
        out = (alpha*values).sum(1)
        return out,aggr_max,alpha


class DotProductAttention(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0,**kwargs):
        super(DotProductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values):
        d = queries.shape[-1]

        scores = torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)

        alpha = self.dropout(torch.softmax(scores,dim=2))
        aggr_max = torch.argmax(alpha, dim=1)

        out = torch.bmm(alpha,values).sum(dim=1)
        return out,aggr_max,alpha


class biLiearityAttention(Attention):

    def __init__(self,key_size,num_hiddens, dropout=0,**kwargs):
        super(biLiearityAttention, self).__init__()


        self.W_q =Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')  # U
        self.W_k =Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')  # V
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):

        queries = self.W_q(queries)  # queries：(N,M,F)
        keys = self.W_k(keys)  # keys:(N,M,F)
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (N,M,M)
        alpha = self.dropout(torch.softmax(scores, dim=2))
        aggr_max = torch.argmax(alpha, dim=1)
        out = torch.bmm(alpha, values).sum(dim=1)
        return out,aggr_max,alpha

class CatAttention(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0,**kwargs):

        super(CatAttention,self).__init__()

        self.W_qk =Linear(key_size + key_size, num_hiddens, bias=False,weight_initializer='glorot')
        self.W_v =Linear(num_hiddens, 1, bias=False,weight_initializer='glorot')
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values):

        scores = self.W_qk(torch.cat((queries,keys),2))

        alpha = torch.softmax(self.W_v(scores),dim=1)
        alpha = self.dropout(alpha)
        aggr_max = torch.argmax(alpha, dim=1)

        out = (alpha * values).sum(1)
        return out, aggr_max,alpha



class Attention_1(nn.Module):
    def __init__(self,key_size,num_hiddens,dropout=0.5,**kwargs):
        super(Attention_1, self).__init__()
        self.score_func = nn.Sequential(
            nn.Linear(key_size,num_hiddens),
            nn.Tanh(),
            nn.Linear(num_hiddens,1,bias=False)
        )

    def forward(self,values):
        w = self.score_func(values)

        alpha = torch.softmax(w,dim=1)
        res = (alpha*values).sum(1)

        return res,alpha

ATTENTIONS={
    'att':At,
    'add':AdditiveAttention,
    'dot':DotProductAttention,
    'bili':biLiearityAttention,
    'cat':CatAttention
}

import numpy as np
if __name__=='__main__':
    N = 1403
    M = 4
    F = 32
    H = 16
    a = np.random.random((N,M,F))

    b = torch.from_numpy(a)

    c = b.float()

    addAttention = AdditiveAttention(key_size=F,query_size=F,num_hiddens=H,dropout=0.2)
    addAttention(c,c,c)

    dotAttention = DotProductAttention(key_size=F,query_size=F,num_hiddens=H,dropout=0.2)
    dotAttention(c,c,c)

    biLiearityAttention=biLiearityAttention(key_size=F,query_size=F,num_hiddens=H,dropout=0.2)
    biLiearityAttention(c,c,c)

    catAttention = CatAttention(key_size=F, query_size=F,num_hiddens=H, dropout=0.2)
    catAttention(c,c,c)