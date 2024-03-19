# _*_codeing=utf-8_*_
# @Time:2022/5/16  20:06
# @Author:mazhixiu
# @File:attentions.py
import torch
"""

"""

import math
import torch
from torch import nn
from torch_geometric.nn import Linear

# 注意力基类
class Attention(nn.Module):
    def __init__(self,**kwargs):
        super(Attention,self).__init__()

    def forward(self,queries,keys,values):
        raise NotImplementedError
"""
Attention:
- 查询向量：聚合结果
- 键值向量：聚合结果
- 评分函数:tanh(Wqv)*Wk
"""
class At(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0.5,**kwargs):
        super(Attention,self).__init__()
        self.score_func = nn.Sequential(
            nn.Linear(key_size,num_hiddens),
            nn.Tanh(),
            nn.Linear(num_hiddens,1,bias=False)
        )
    def forward(self,queries,keys,values):
        # z:(N,M,F)
        #(N,M,F)*(F,H)->(N,M,H)->(N,M,1)
        # print(z.shape)# torch.Size([2708, 4, 256])
        w = self.score_func(values)

        # print(w.shape)# torch.Size([2708, 4, 1])
        alpha = torch.softmax(w,dim=1)
        # print(alpha.shape)
        """
        计算哪个聚合器更重要,形成直方图分布，统计哪个聚合器对节点聚合比较重要
        """
        aggr_max = torch.argmax(alpha,dim=1)
        # print(aggr_max)

        res = (alpha*values).sum(1)  # 这里可以再变化

        # print(res.shape)# torch.Size([2708,256])
        return res,aggr_max,alpha

"""
加性attention:
- 查询向量：聚合结果经过一个线性变换
- 键值向量：聚合结果经过一个线性变换
- 评分函数：s = WqQ*WkK
"""
class AdditiveAttention(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0,**kwargs):
        super(AdditiveAttention,self).__init__(**kwargs)
        # self.W_k = nn.Linear(key_size,num_hiddens,bias=False)
        # self.W_q = nn.Linear(key_size,num_hiddens,bias=False)
        # self.W_v = nn.Linear(num_hiddens,1,bias=False)
        self.W_k = Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')
        self.W_q = Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')
        self.W_v = Linear(num_hiddens, 1, bias=False,weight_initializer='glorot')
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values):
        #queries:(N,M,F)->(N,M,H） #keys:(N,M,F)->(N,M,H)
        # print(queries.shape)#torch.Size([2485, 2, 128])
        # print(keys.shape)
        # print(values.shape)
        # print(self.W_q)# torch.Size([2006, 4, 128])
        queries,keys = self.W_q(queries),self.W_k(keys)
        # print(queries.shape)
        # print(keys.shape)
        # features:(N,M,H)
        features = torch.tanh(queries+keys)
        # print(features.shape)
        # scores:(N,M,H)->(N,M,1)
        scores = self.W_v(features)
        # alpha:聚合器的不同权重。维度1上的数据用于计算注意力权重分布
        alpha = torch.softmax(scores,dim=1)
        alpha = self.dropout(alpha)
        # aggr_max:哪个聚合器最重要。可视化注意力权重分布需要的数据。
        aggr_max = torch.argmax(alpha, dim=1)
        #(N,M,1)*(N,M,F)广播乘=>(N,M,F)=>(N,F)
        out = (alpha*values).sum(1)
        return out,aggr_max,alpha

"""
缩放点积：
- 查询向量：聚合结果
- 键值向量: 聚合结果
- 评分函数：Q*K/sqrt(d)
"""
class DotProductAttention(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0,**kwargs):
        super(DotProductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values):
        d = queries.shape[-1]
        #(N,M,F)*(N,F,M)=>(N,M,M)
        # 点积运算
        scores = torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
        # 维度2上的数据用作注意力权重分配的计算
        alpha = self.dropout(torch.softmax(scores,dim=2))
        aggr_max = torch.argmax(alpha, dim=1)
        # (N,M,M)*(N,M,F) =>(N,M,F)(维度2上的数据是分配给不同聚合器的注意力权重)
        out = torch.bmm(alpha,values).sum(dim=1)
        return out,aggr_max,alpha

"""
双线性：
- 查询向量：聚合结果经过一个线性变换
- 键值向量：聚合结果经过一个线性变换
- 评分函数：s = WqQ*WkK
"""
class biLiearityAttention(Attention):

    def __init__(self,key_size,num_hiddens, dropout=0,**kwargs):
        super(biLiearityAttention, self).__init__()

        # self.W_q = nn.Linear(key_size, num_hiddens, bias=False) # U
        # self.W_k = nn.Linear(key_size, num_hiddens, bias=False) # V
        self.W_q =Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')  # U
        self.W_k =Linear(key_size, num_hiddens, bias=False,weight_initializer='glorot')  # V
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):

        # 相比于点积模型，引入非对称性。
        queries = self.W_q(queries)  # queries：(N,M,F)
        keys = self.W_k(keys)  # keys:(N,M,F)
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (N,M,M)
        alpha = self.dropout(torch.softmax(scores, dim=2))
        aggr_max = torch.argmax(alpha, dim=1)
        out = torch.bmm(alpha, values).sum(dim=1)
        return out,aggr_max,alpha

"""
拼接模型：
"""
class CatAttention(Attention):
    def __init__(self,key_size,num_hiddens,dropout=0,**kwargs):

        super(CatAttention,self).__init__()

        # self.W_qk = nn.Linear(key_size+key_size,num_hiddens,bias=False)
        # self.W_v = nn.Linear(num_hiddens,1,bias=False)
        self.W_qk =Linear(key_size + key_size, num_hiddens, bias=False,weight_initializer='glorot')
        self.W_v =Linear(num_hiddens, 1, bias=False,weight_initializer='glorot')
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values):
        # (N,M,F)cat(N,M,F)=>(N,M,2F)
        # (N,M,2F)*(2F,H)=>(N,M,H)
        scores = self.W_qk(torch.cat((queries,keys),2))
        # (N,M,H)->(N,M,1)
        alpha = torch.softmax(self.W_v(scores),dim=1)
        alpha = self.dropout(alpha)
        aggr_max = torch.argmax(alpha, dim=1)
        # (N,M,1)*(N,M,F)广播乘=>(N,M,F)=>(N,F)
        out = (alpha * values).sum(1)
        # print(out.shape)
        return out, aggr_max,alpha

# class MultiHeadAttention(nn.Module):
#     # 多头注意力
#     def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
#         super(MultiHeadAttention,self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.attention = DotProductAttention(dropout)
#         self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
#         self.W_k = nn.Linear(value_size,num_hiddens,bias=bias)
#         self.W_k = nn.Linear(value_size,num_hiddens,bias=bias)
#         self.W_o = nn.Linear(num_hiddens,num_heads,bias=bias)

class Attention_1(nn.Module):
    def __init__(self,key_size,num_hiddens,dropout=0.5,**kwargs):
        super(Attention_1, self).__init__()
        self.score_func = nn.Sequential(
            nn.Linear(key_size,num_hiddens),
            nn.Tanh(),
            nn.Linear(num_hiddens,1,bias=False)
        )
    # def
    def forward(self,values):
        w = self.score_func(values)
        # print(w.shape)# torch.Size([2485, 2, 1])
        alpha = torch.softmax(w,dim=1)
        # print(alpha.shape)

        res = (alpha*values).sum(1)  # 这里可以再变化

        # print(res.shape)# torch.Size([2485, 128])
        return res,alpha

ATTENTIONS={
    'att':At,
    'add':AdditiveAttention,
    'dot':DotProductAttention,
    'bili':biLiearityAttention,
    'cat':CatAttention
}

import numpy as np

# 注意力测试
if __name__=='__main__':
    N = 1403
    M = 4
    F = 32
    H = 16
    a = np.random.random((N,M,F))

    b = torch.from_numpy(a)
    # c = torch.Tensor(b)
    # print(b.shape)
    c = b.float()
    # # print(c)
    # attention = Attention(F,F,H)
    # attention(c,c,c)
    # __init__(self,key_size,query_size,num_hiddens,dropout=0,**kwargs)
    addAttention = AdditiveAttention(key_size=F,query_size=F,num_hiddens=H,dropout=0.2)
    addAttention(c,c,c)

    # (self,key_size,query_size, num_hiddens,dropout=0.2,**kwargs)
    dotAttention = DotProductAttention(key_size=F,query_size=F,num_hiddens=H,dropout=0.2)
    dotAttention(c,c,c)
    #
    #(self,key_size,query_size, num_hiddens, dropout=0,**kwargs)
    biLiearityAttention=biLiearityAttention(key_size=F,query_size=F,num_hiddens=H,dropout=0.2)
    biLiearityAttention(c,c,c)
    # (self,query_size,key_size,num_hiddens,dropout=0,**kwargs):
    catAttention = CatAttention(key_size=F, query_size=F,num_hiddens=H, dropout=0.2)
    catAttention(c,c,c)