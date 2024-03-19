# _*_codeing=utf-8_*_
# @Time:2022/3/22  11:43

# @File:aggregators.py
import torch
from torch import Tensor
from torch_scatter import scatter
from typing import Optional
from torch_geometric.utils import to_dense_batch


def aggregate_sum(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='sum')


def aggregate_mean(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='mean')


def aggregate_min(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='min')


def aggregate_max(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='max')


def aggregate_median(src:Tensor,index:Tensor,dim_size:Optional[int]):

    ix = torch.argsort(index)
    index = index[ix]
    src = src[ix]
    dense_x,mask = to_dense_batch(src,index)
    out = src.new_zeros(dense_x.size(0),dense_x.size(-1))
    deg = mask.sum(dim=1)

    dense_x = torch.squeeze(dense_x)

    for i in deg.unique():
        deg_mask = deg==i
        out[deg_mask] = dense_x[deg_mask,:i].median(dim=1).values
    return out


def aggregate_mode(src:Tensor,index:Tensor,dim_size:Optional[int]):

    ix = torch.argsort(index)
    index = index[ix]
    src = src[ix]

    dense_x, mask = to_dense_batch(src, index)
    out = src.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    dense_x = torch.squeeze(dense_x)

    for i in deg.unique():
        deg_mask = deg == i
        out[deg_mask] = dense_x[deg_mask, :i].mode(dim=1).values

    return out


def aggregate_trimmed(src:Tensor,index:Tensor,dim_size:Optional[int]):
    ix = torch.argsort(index)
    index = index[ix]
    src = src[ix]

    dense_x, mask = to_dense_batch(src, index)
    out = src.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    dense_x = torch.squeeze(dense_x)

    for i in deg.unique():
        deg_mask = deg == i

        if i == 1 or i == 2:

            out[deg_mask] = dense_x[deg_mask, :i].mean(dim=1)
        else:

            min = dense_x[deg_mask, :i].min(dim=1)  # 2
            min_indice = min.indices.mode().values
            dense_x[deg_mask, :i].index_select(1, min_indice).fill_(0)
            max = dense_x[deg_mask, :i].max(dim=1)
            max_indice = max.indices.mode().values
            dense_x[deg_mask, :i].index_select(1, max_indice).fill_(0)

            out[deg_mask] = dense_x[deg_mask, :i].mean(dim=1)

    return out


#求方差
def aggregate_var(src,index,dim_size):
    mean = aggregate_mean(src,index,dim_size)
    mean_squares = aggregate_mean(src*src,index,dim_size)
    return mean_squares - mean*mean

def aggregate_std(src,index,dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src,index,dim_size))+1e-5)

AGGREGATORS = {
    'sum':aggregate_sum,
    'add':aggregate_sum,
    'mean':aggregate_mean,
    'min':aggregate_min,
    'max':aggregate_max,
    'median':aggregate_median,
    'mode':aggregate_mode,
    'trimmed':aggregate_trimmed,
    'var':aggregate_var,
    'std':aggregate_std
}

