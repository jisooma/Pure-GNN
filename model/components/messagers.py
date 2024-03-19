# _*_codeing=utf-8_*_
# @Time:2022/11/14  16:04
# @Author:mazhixiu
# @File:messagers.py

import torch
from torch import Tensor

from torch_geometric.typing import  OptTensor
from typing import Optional
from torch.nn import functional as F
from torch_geometric.utils import softmax
# from

"""
经典图神经网络的三个消息函数
"""
def message_const(x_j: Tensor,):
    return x_j

def message_gcn(x_j: Tensor,edge_weight: OptTensor,) -> Tensor:
    return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

# pyg实现
def message_gat(
            x_j: Tensor,
            alpha_j: Tensor, alpha_i: OptTensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            ) -> Tensor:
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    # negative_slope=0.2
    alpha = F.leaky_relu(alpha, 0.2)
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j*alpha

# cosine,euclidean,correlation,chebyshev,braycurtis,canberra,cityblock,sqeuclidean
def message_euclidean(beta,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int],):
    i_j = F.normalize((x_i - x_j), p=2, dim=-1)

    alpha = beta*i_j.sum(dim=-1)

    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j
    return x_j, alpha


def message_chebyshev(beta,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int],):
    i_j_max = torch.max(torch.abs(x_i - x_j),dim=1)

    alpha = beta * i_j_max.sum(dim=-1)

    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j
    return x_j, alpha


def message_braycurtis(beta,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    # pass
    l1_diff = torch.abs(x_i-x_j)
    l1_sum = torch.abs(x_i+x_j)

    alpha = beta * torch.div((l1_diff.sum(dim=1)),(l1_sum.sum(dim=1)))

    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j

    return x_j, alpha

def message_canberra(beta,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    # pass
    abs_i_j = torch.abs(x_i - x_j)
    abs_i = torch.abs(x_i)
    abs_j = torch.abs(x_j)

    alpha = beta * torch.div(abs_i_j, (abs_j+abs_i))

    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j

    return x_j, alpha

def message_cityblock(beta,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    abs_i_j = torch.abs(x_i - x_j)
    alpha = beta * abs_i_j.sum(dim=1)

    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j

    return x_j, alpha

def message_sqeuclidean(beta,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    i_j = x_i - x_j

    i_j_dot = torch.dot(x_i,x_j)
    alpha = beta * i_j.sum(dim=-1)

    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j
    return x_j, alpha


"""
为了抵御对抗样本攻击设计的消息函数,我们基于同质性原则设计的消息函数,与一些基于特征的预处理方案不同,我们的方案是在模型卷积层的消息生成阶段
基于结构的GCNSVD预处理,不能以消息生成函数的处理特征的方式进行处理.
针对异质图,可以用结构相似度
"""
def message_jaccard_similiarity(beta,x_j: Tensor, alpha_i: Tensor,index, ptr, size_i,):
    x_norm = beta*alpha_i
    # print(beta.shape)
    # print(alpha_i.shape)
    # print(x_norm.shape)
    alpha = softmax(x_norm, index, ptr, size_i)
    # print(alpha.shape)# torch.Size([12623])
    # print(x_j.shape)# torch.Size([12623, 128])
    return alpha.view(-1, 1)*x_j

def message_cosine_similiarity(beta,x_j: Tensor,
                x_norm_i: Tensor, x_norm_j: Tensor,index: Tensor,ptr: OptTensor,
                size_i=Optional[int],
                ):
    # print((x_norm_i * x_norm_j).sum(dim=-1).shape)
    alpha = beta * (x_norm_i * x_norm_j).sum(dim=-1)
    # print(alpha.shape)
    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j
    return x_j,alpha


def message_cosine_similiarity_1(
        # beta,
                                 # x_ii:Tensor,
                                 # x_jj: Tensor,
                                x_j: Tensor,
                x_norm_i: Tensor, x_norm_j: Tensor,index: Tensor,ptr: OptTensor,
                size_i=Optional[int],
                ):
    # print(x_j.shape)
    # print(x_i.shape)
    # print(x_norm_i.shape)
    # print(x_norm_j.shape)
    # print((x_norm_i * x_norm_j).sum(dim=-1).shape)
    # prin
    # print(type(x_ii))
    # print(type(x_jj))
    # print(type(x_norm_i))
    # print(type(x_norm_j))
    # alpha = (beta*((x_ii*x_jj))/(beta*(x_norm_i * x_norm_j))).sum(dim=-1)
    alpha = (x_norm_i * x_norm_j).sum(dim=-1)
    # alpha = (beta * ((x_ii * x_jj))).sum(dim=-1)
    # alpha = ((x_ii * x_jj)).sum(dim=-1)
    # print(alpha.shape)
    alpha = softmax(alpha, index, ptr, size_i)
    # print(alpha.view(-1, 1).shape)
    x_j = alpha.view(-1, 1) * x_j
    return x_j,alpha

# 可以设计其他更加健壮的消息聚合函数
def messsage_others_similarity():
    pass


"""
以下三个是GAT消息函数的变体
"""
def message_cos(x_j: Tensor,alpha_j: Tensor, alpha_i: OptTensor,index, ptr, size_i,):
    x_norm = alpha_i+alpha_j
    alpha = softmax(x_norm, index, ptr, size_i)
    return x_j*alpha

def message_linear(x_j: Tensor,
            alpha_j: Tensor, alpha_i: OptTensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            *kwargs
           ):
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    alpha = F.tanh(input=alpha)
    alpha = softmax(alpha, index, ptr, size_i)
    # alpha = F.dropout(alpha, p=0, training=self.training)
    return x_j * alpha

def message_gen_linear(w,x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,index, ptr, size_i,*kwargs):
    # 这里的w
    alpha =x_norm_i+x_norm_j
    alpha = F.tanh(input=alpha)
    alpha = w * alpha
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j * alpha

def message_gat_1(
            x_j: Tensor,
            alpha_j: Tensor, alpha_i: OptTensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            ) -> Tensor:
    alpha = torch.cat((alpha_i,alpha_j),dim=1)
    # negative_slope=0.2
    alpha = F.leaky_relu(alpha, 0.2)
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j*alpha


def message_gat_2(
            a,#2F'
            x_i:Tensor,
            x_j: Tensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            *kwargs
            ) -> Tensor:

    alpha = a*torch.cat((x_i,x_j),dim=0)
    # negative_slope=0.2
    alpha = F.leaky_relu(alpha, 0.2)
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j*alpha

def message_gat_3(
            a,#F'
            x_i:Tensor,
            x_j: Tensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            *kwargs
            ) -> Tensor:

    alpha = a*(x_i+x_j)
    # negative_slope=0.2
    alpha = F.leaky_relu(alpha, 0.2)
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j*alpha

def message_linear_1(x_j: Tensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int], *kwargs
           ):
    alpha = torch.sum(x_j,dim=0)
    alpha = F.tanh(input=alpha)
    alpha = softmax(alpha, index, ptr, size_i)
    # alpha = F.dropout(alpha, p=0, training=self.training)
    return x_j * alpha.unsqueeze(-1)


def message_gen_linear_1(w,x_i:Tensor,x_j: Tensor, index, ptr, size_i,*kwargs):
    alpha =x_i+x_j
    alpha = F.tanh(input=alpha)
    alpha = w * alpha
    alpha = softmax(alpha, index, ptr, size_i)

    return x_j * alpha


"""
gnnguard,rgcn,cure_gnn等防御方案有一部分也是修改的消息函数(rgcn多了目标函数的修改,cure_gnn多了特征重构部分,gnnguard多了删边的部分)
"""
def message_gnn_guard():
    pass
def message_rgcn():
    pass
def message_cure_gnn():
    pass


"""

"""

MESSAGERS = {

    'message_const':message_const,
    'gcn':message_gcn,
    'const':message_const,

    'gat': message_gat,
    'cos':message_cos,
    'linear': message_linear,
    'gen_linear':message_gen_linear,


    'jac_sim': message_jaccard_similiarity,
    'cos_sim': message_cosine_similiarity,
    'cos_sim_1': message_cosine_similiarity_1,
    # 'gat_1': message_gat_1,
    # 'gat_2': message_gat_2,
    # 'gat_3': message_gat_3,

    # 'gen_linear_1': message_gen_linear_1,
    # 'linear_1': message_linear_1,
}


