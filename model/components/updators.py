# _*_codeing=utf-8_*_
# @Time:2022/11/14  16:04
# @Author:mazhixiu
# @File:updators.py

import torch
from torch import Tensor
import torch.nn.functional as F


# 当前层更新节点特征的时候，把自身节点特征加上
def update_const(inputs: Tensor,)->Tensor:
    return inputs

def update_const_1(inputs: Tensor,x_i)->Tensor:
    return inputs + x_i

# 当前层更新节点特征的时候，把自身节点特征加上
def update_pwd_layer_x(inputs: Tensor,attr,x_i,) -> Tensor:
    attr = F.sigmoid(attr)
    inputs = inputs * (1 - attr) + x_i * attr
    return inputs

# 当前层更新节点的时候，进行密集特征过滤
def update_pwd_layer_x_dense_filter(inputs: Tensor,lin,x_i)->Tensor:
    gates = torch.cat([inputs,x_i],dim=1)
    # print(gate/s.shape)# torch.Size([19717, 256])  # torch.Size([2485, 14])
    # print(lin)
    gates = lin(gates)#
    # print(gates.shape)# torch.Size([19717, 128])
    return torch.sigmoid(gates)*inputs

def update_pwd_layer_x_feature_dense_filter(inputs: Tensor,x_i,a)->Tensor:
    gates = torch.add(inputs,x_i)
    # print(gates.shape)#  torch.Size([19717, 3])
    gates = a(gates)#
    # print(gates.shape)#torch.Size([19717, 1])
    return torch.sigmoid(gates)*inputs

def update_pwd_layer_x_feature_sparse_filter(inputs: Tensor,x_i,a)->Tensor:
    gates = a*torch.add(inputs,x_i)
    return torch.sigmoid(gates)*inputs

# 当前层更新节点特征的时候，进行稀疏特征过滤
def update_pwd_layer_x_sparse_filter(inputs: Tensor,lin,x_i,a)->Tensor:
    gates = torch.cat([inputs,x_i],dim=1)
    gates = torch.relu(lin(gates))
    # print(gates.shape)#  torch.Size([19717, 3])
    gates = a(gates)#
    # print(gates.shape)#torch.Size([19717, 1])
    return torch.sigmoid(gates)*inputs

# linear
def update_pwd_layer_x_linear(inputs: Tensor,w_s,w_n,x_i,)->Tensor:
    inputs = w_s(x_i)+w_n(inputs)
    inputs= torch.relu(inputs)
    return inputs

# concat
def update_pwd_layer_x_concat(inputs: Tensor,w_s,w_n,x_i)->Tensor:
    inputs = torch.relu(w_s(x_i) + w_n (inputs))
    return torch.cat((inputs,x_i),dim=1)

# interpolation
def update_pwd_layer_x_interpolation(inputs: Tensor,w_s,w_n,x_i,a_1,a_2)->Tensor:
    inputs = torch.relu(w_s(x_i) + w_n(inputs))
    inputs = a_1*inputs+a_2*x_i
    return inputs


"""
以下在模型中，不在每一层中
"""
# 每一层在更新节点的时候，把上一层的节点加上
def update_previous_layer_x(inputs: Tensor,attr,x_p,):

    if attr!=None:
        inputs = inputs * (1 - attr) + x_p * attr
    else:
        inputs += x_p
    return inputs

# 每一层在更新节点的时候，把之前所有层的节点特征加上
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
