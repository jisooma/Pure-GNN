# _*_codeing=utf-8_*_
# @Time:2022/11/14  16:04

# @File:updators.py

import torch
from torch import Tensor
import torch.nn.functional as F



def update_const(inputs: Tensor,)->Tensor:
    return inputs

def update_const_1(inputs: Tensor,x_i)->Tensor:
    return inputs + x_i


def update_pwd_layer_x(inputs: Tensor,attr,x_i,) -> Tensor:
    attr = F.sigmoid(attr)
    inputs = inputs * (1 - attr) + x_i * attr
    return inputs


def update_pwd_layer_x_dense_filter(inputs: Tensor,lin,x_i)->Tensor:
    gates = torch.cat([inputs,x_i],dim=1)
    gates = lin(gates)#
    return torch.sigmoid(gates)*inputs

def update_pwd_layer_x_feature_dense_filter(inputs: Tensor,x_i,a)->Tensor:
    gates = torch.add(inputs,x_i)

    gates = a(gates)
    return torch.sigmoid(gates)*inputs

def update_pwd_layer_x_feature_sparse_filter(inputs: Tensor,x_i,a)->Tensor:
    gates = a*torch.add(inputs,x_i)
    return torch.sigmoid(gates)*inputs


def update_pwd_layer_x_sparse_filter(inputs: Tensor,lin,x_i,a)->Tensor:
    gates = torch.cat([inputs,x_i],dim=1)
    gates = torch.relu(lin(gates))
    gates = a(gates)#
    return torch.sigmoid(gates)*inputs


def update_pwd_layer_x_linear(inputs: Tensor,w_s,w_n,x_i,)->Tensor:
    inputs = w_s(x_i)+w_n(inputs)
    inputs= torch.relu(inputs)
    return inputs


def update_pwd_layer_x_concat(inputs: Tensor,w_s,w_n,x_i)->Tensor:
    inputs = torch.relu(w_s(x_i) + w_n (inputs))
    return torch.cat((inputs,x_i),dim=1)


def update_pwd_layer_x_interpolation(inputs: Tensor,w_s,w_n,x_i,a_1,a_2)->Tensor:
    inputs = torch.relu(w_s(x_i) + w_n(inputs))
    inputs = a_1*inputs+a_2*x_i
    return inputs



def update_previous_layer_x(inputs: Tensor,attr,x_p,):

    if attr!=None:
        inputs = inputs * (1 - attr) + x_p * attr
    else:
        inputs += x_p
    return inputs


def update_all_layers_x(inputs: Tensor,att_type,x_is,):
    if att_type == 'cat':
        return torch.cat(x_is, dim=-1)
    elif att_type == 'max':
        return torch.stack(x_is, dim=-1).max(dim=-1)[0]
    # elif att_type == 'lstm':
    #     x = torch.stack(x_is, dim=1)  # [num_nodes, num_layers, num_channels]
    #     alpha, _ = self.lstm(x)
    #     alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
    #     alpha = torch.softmax(alpha, dim=-1)
    #     return (x * alpha.unsqueeze(-1)).sum(dim=1)
    return None

UPDATORS={
    'const':update_const,
    'const_1':update_const_1,
    'initial_x':update_pwd_layer_x,
    'feature_dense_filter':update_pwd_layer_x_feature_dense_filter,
    'feature_sparse_filter':update_pwd_layer_x_feature_sparse_filter,
    'dense_filter':update_pwd_layer_x_dense_filter,
    'sparse_filter':update_pwd_layer_x_sparse_filter,
    'linear_x':update_pwd_layer_x_linear,
    'concat_x':update_pwd_layer_x_concat,
    'interpolation_x':update_pwd_layer_x_interpolation,
    # 'previous_layer_x':update_previous_layer_x,
    # 'all_layers_x':update_all_layers_x,
}
