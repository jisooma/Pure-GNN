import torch
from torch import Tensor

from torch_geometric.typing import  OptTensor
from typing import Optional
from torch.nn import functional as F
from torch_geometric.utils import softmax

def message_const(x_j: Tensor,):
    return x_j

def message_gcn(x_j: Tensor,edge_weight: OptTensor,) -> Tensor:
    return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

def message_gat(
            x_j: Tensor,
            alpha_j: Tensor, alpha_i: OptTensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            ) -> Tensor:
    alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
    alpha = F.leaky_relu(alpha, 0.2)
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j*alpha


def message_euclidean(beta,x_i: Tensor,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int],):
    i_j = F.normalize((x_i - x_j), p=2, dim=-1)
    alpha = beta*i_j.sum(dim=-1)
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j
    return x_j, alpha


def message_chebyshev(beta,x_i: Tensor,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int],):
    i_j_max = torch.max(torch.abs(x_i - x_j),dim=1)
    alpha = beta * i_j_max.sum(dim=-1)
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j
    return x_j, alpha


def message_braycurtis(beta,x_i: Tensor,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    l1_diff = torch.abs(x_i-x_j)
    l1_sum = torch.abs(x_i+x_j)
    alpha = beta * torch.div((l1_diff.sum(dim=1)),(l1_sum.sum(dim=1)))
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j

    return x_j, alpha

def message_canberra(beta,x_i: Tensor,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    abs_i_j = torch.abs(x_i - x_j)
    abs_i = torch.abs(x_i)
    abs_j = torch.abs(x_j)
    alpha = beta * torch.div(abs_i_j, (abs_j+abs_i))
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j

    return x_j, alpha

def message_cityblock(beta,x_i: Tensor,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    abs_i_j = torch.abs(x_i - x_j)
    alpha = beta * abs_i_j.sum(dim=1)
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j

    return x_j, alpha

def message_sqeuclidean(beta,x_i: Tensor,x_j: Tensor,
                index: Tensor,ptr: OptTensor,
                size_i=Optional[int]):
    i_j = x_i - x_j
    i_j_dot = torch.dot(x_i,x_j)
    alpha = beta * i_j.sum(dim=-1)
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j
    return x_j, alpha


def message_jaccard_similiarity(beta,x_j: Tensor, alpha_i: Tensor,index, ptr, size_i,):
    x_norm = beta*alpha_i
    alpha = softmax(x_norm, index, ptr, size_i)
    return alpha.view(-1, 1)*x_j

def message_cosine_similiarity(beta,x_j: Tensor,
                x_norm_i: Tensor, x_norm_j: Tensor,index: Tensor,ptr: OptTensor,
                size_i=Optional[int],
                ):

    alpha = beta * (x_norm_i * x_norm_j).sum(dim=-1)
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j
    return x_j,alpha


def message_cosine_similiarity_1(
                                x_j: Tensor,
                x_norm_i: Tensor, x_norm_j: Tensor,index: Tensor,ptr: OptTensor,
                size_i=Optional[int],
                ):

    alpha = (x_norm_i * x_norm_j).sum(dim=-1)
    alpha = softmax(alpha, index, ptr, size_i)
    x_j = alpha.view(-1, 1) * x_j
    return x_j,alpha


def messsage_others_similarity():
    pass

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
    return x_j * alpha

def message_gen_linear(w,x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,index, ptr, size_i,*kwargs):
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
    alpha = F.leaky_relu(alpha, 0.2)
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j*alpha


def message_gat_2(
            a,
            x_i:Tensor,
            x_j: Tensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            *kwargs
            ) -> Tensor:

    alpha = a*torch.cat((x_i,x_j),dim=0)
    alpha = F.leaky_relu(alpha, 0.2)
    alpha = softmax(alpha, index, ptr, size_i)
    return x_j*alpha

def message_gat_3(
            a,
            x_i:Tensor,
            x_j: Tensor,
            index: Tensor, ptr: OptTensor,
            size_i: Optional[int],
            *kwargs
            ) -> Tensor:

    alpha = a*(x_i+x_j)
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
    return x_j * alpha.unsqueeze(-1)


def message_gen_linear_1(w,x_i:Tensor,x_j: Tensor, index, ptr, size_i,*kwargs):
    alpha =x_i+x_j
    alpha = F.tanh(input=alpha)
    alpha = w * alpha
    alpha = softmax(alpha, index, ptr, size_i)

    return x_j * alpha


def message_gnn_guard():
    pass
def message_rgcn():
    pass
def message_cure_gnn():
    pass


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
}


