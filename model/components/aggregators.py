# _*_codeing=utf-8_*_
# @Time:2022/3/22  11:43
# @Author:mazhixiu
# @File:aggregators.py
import torch
from torch import Tensor
from torch_scatter import scatter
from typing import Optional
from torch_geometric.utils import to_dense_batch
"""
应该需要做pre实验，判断扰动边带来的边，到底是为中心节点带来的是max还是min。
后面使用注意力机制的话，也需要看看注意力为这写聚合器所聚合结果分配的权重是多少？
理想情况是，trimmed、median、mean、mode这些应该高一些。
"""
# 求和
def aggregate_sum(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='sum')

# 求均值，能够对扰动边带来的边进行稀释
def aggregate_mean(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='mean')

# 求最小值，非常有可能把扰动值删掉，也非常有可能只把扰动值留下
def aggregate_min(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='min')

# 求max,非常有可能把扰动值删掉，也非常有可能只把扰动值留下
def aggregate_max(src:Tensor,index:Tensor,dim_size:Optional[int]):
    return scatter(src,index,0,None,dim_size,reduce='max')

"""
mediangcn中健壮的聚合函数
"""
# 求中位数，这样容易把max和min带来的扰动值删掉。
def aggregate_median(src:Tensor,index:Tensor,dim_size:Optional[int]):
    # ix对应节点按序排列的位置索引
    ix = torch.argsort(index)
    # 根据顺序把节点进行重新排序
    index = index[ix]
    # 按顺序也把特征矩阵进行排列
    # 排好序的特征可以看作是一个（N0+N1+N2+..+Nnum_nodes）*F的特征矩阵
    # num_nodes为节点个数。N0代表节点0（也就是与节点0相邻的节点个数有多少个）
    src = src[ix]
    """
    给定节点特征X∈R(N1+N2+NB)*F
    创建一个B*Nmax*F的dense特征tensor
    B：所有节点个数
    Nmax：代表邻居数最多的那个节点对应的邻居数
    F:节点特征维度
    mask:进行掩码
    """
    dense_x,mask = to_dense_batch(src,index)
    # print(dense_x.shape)
    """
    torch.Size([2485, 169, 1, 128])
    torch.Size([2485, 169])
    """
    # print(dense_x.shape)# torch.Size([2708, 168, 1433])
    # print(mask.shape) # torch.Size([2708, 168])
    out = src.new_zeros(dense_x.size(0),dense_x.size(-1))
    deg = mask.sum(dim=1)
    """
    torch.Size([2485])
    torch.Size([2485, 128])
    """
    # print(deg.shape)#torch.Size([2708])
    # print(out.shape)
    dense_x = torch.squeeze(dense_x)
    # print(dense_x.shape)
    for i in deg.unique():
        deg_mask = deg==i
        out[deg_mask] = dense_x[deg_mask,:i].median(dim=1).values
    return out

# 求众数，少数min，max的值可能插入扰动边对应的节点引入的。
def aggregate_mode(src:Tensor,index:Tensor,dim_size:Optional[int]):

    ix = torch.argsort(index)
    index = index[ix]
    src = src[ix]

    dense_x, mask = to_dense_batch(src, index)
    out = src.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    dense_x = torch.squeeze(dense_x)
    # print(dense_x.shape)
    for i in deg.unique():
        deg_mask = deg == i
        out[deg_mask] = dense_x[deg_mask, :i].mode(dim=1).values

    return out

# 剪枝,剪掉最大的和最小的,然后再求和，因为最小的和最大的可能是扰动边加入的，
# 但是，相比于中位数，保留了正常邻居更多的信息
# 也可以
def aggregate_trimmed(src:Tensor,index:Tensor,dim_size:Optional[int]):
    ix = torch.argsort(index)
    index = index[ix]
    src = src[ix]

    dense_x, mask = to_dense_batch(src, index)
    out = src.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    dense_x = torch.squeeze(dense_x)
    # print(dense_x.shape)
    for i in deg.unique():
        deg_mask = deg == i
        # 如果节点之后只有一个邻居或者只有2个邻居，还是求均值.防止出现全部删为0，容易出现问题。
        # 而且，一般扰动都是不明显的1邻居，，本来只有一个再加一个邻居就太明显了，一般都是给度数比较高的节点添加，可以做实验验证
        if i == 1 or i == 2:
            # print(dense_x[deg_mask, :i].mean(dim=1))
            out[deg_mask] = dense_x[deg_mask, :i].mean(dim=1)
        else:
            # print('else')
            # print(dense_x[deg_mask,:i].shape)#torch.Size([583, 3, 16])
            # 找到min最多的那一邻居，删掉(将min最多的那一个邻居特征置为0)
            min = dense_x[deg_mask, :i].min(dim=1)  # 2
            # 对min.indices进行处理
            min_indice = min.indices.mode().values
            # print(min.indices.mode().values)

            dense_x[deg_mask, :i].index_select(1, min_indice).fill_(0)
            # 找到max最多的那一邻居，删掉(将max最多的那一个邻居特征置为0)

            max = dense_x[deg_mask, :i].max(dim=1)
            max_indice = max.indices.mode().values
            dense_x[deg_mask, :i].index_select(1, max_indice).fill_(0)

            out[deg_mask] = dense_x[deg_mask, :i].mean(dim=1)

    return out
# # 求中位数，这样容易把max和min带来的扰动值删掉。
# def aggregate_median(src:Tensor,index:Tensor,dim_size:Optional[int]):
#     # ix对应节点按序排列的位置索引
#     ix = torch.argsort(index)
#     # 根据顺序把节点进行重新排序
#     index = index[ix]
#     # 按顺序也把特征矩阵进行排列
#     # 排好序的特征可以看作是一个（N0+N1+N2+..+Nnum_nodes）*F的特征矩阵
#     # num_nodes为节点个数。N0代表节点0（也就是与节点0相邻的节点个数有多少个）
#     src = src[ix]
#     """
#     给定节点特征X∈R(N1+N2+NB)*F
#     创建一个B*Nmax*F的dense特征tensor
#     B：所有节点个数
#     Nmax：代表邻居数最多的那个节点对应的邻居数
#     F:节点特征维度
#     mask:进行掩码
#     """
#     dense_x,mask = to_dense_batch(src,index)
#     # print(dense_x.shape)
#     """
#     torch.Size([2485, 169, 1, 128])
#     torch.Size([2485, 169])
#     """
#     # print(dense_x.shape)# torch.Size([2708, 168, 1433])
#     # print(mask.shape) # torch.Size([2708, 168])
#     out = src.new_zeros(dense_x.size(0),dense_x.size(-1))
#     deg = mask.sum(dim=1)
#     """
#     torch.Size([2485])
#     torch.Size([2485, 128])
#     """
#     # print(deg.shape)#torch.Size([2708])
#     # print(out.shape)
#     dense_x = torch.squeeze(dense_x)
#     # print(dense_x.shape)
#     for i in deg.unique():
#         deg_mask = deg==i
#         out[deg_mask] = dense_x[deg_mask,:i].median(dim=1).values
#     return out
#
# # 求众数，少数min，max的值可能插入扰动边对应的节点引入的。
# def aggregate_mode(src:Tensor,index:Tensor,dim_size:Optional[int]):
#
#     ix = torch.argsort(index)
#     index = index[ix]
#     src = src[ix]
#
#     dense_x, mask = to_dense_batch(src, index)
#     out = src.new_zeros(dense_x.size(0), dense_x.size(-1))
#     deg = mask.sum(dim=1)
#     dense_x = torch.squeeze(dense_x)
#     # print(dense_x.shape)
#     for i in deg.unique():
#         deg_mask = deg == i
#         out[deg_mask] = dense_x[deg_mask, :i].mode(dim=1).values
#
#     return out
#
# # 剪枝,剪掉最大的和最小的,然后再求和，因为最小的和最大的可能是扰动边加入的，
# # 但是，相比于中位数，保留了正常邻居更多的信息
# # 也可以
# def aggregate_trimmed(src:Tensor,index:Tensor,dim_size:Optional[int]):
#     ix = torch.argsort(index)
#     index = index[ix]
#     src = src[ix]
#
#     dense_x, mask = to_dense_batch(src, index)
#     out = src.new_zeros(dense_x.size(0), dense_x.size(-1))
#     deg = mask.sum(dim=1)
#     dense_x = torch.squeeze(dense_x)
#     # print(dense_x.shape)
#     for i in deg.unique():
#         deg_mask = deg == i
#         # 如果节点之后只有一个邻居或者只有2个邻居，还是求均值.防止出现全部删为0，容易出现问题。
#         # 而且，一般扰动都是不明显的1邻居，，本来只有一个再加一个邻居就太明显了，一般都是给度数比较高的节点添加，可以做实验验证
#         if i == 1 or i == 2:
#             # print(dense_x[deg_mask, :i].mean(dim=1))
#             out[deg_mask] = dense_x[deg_mask, :i].mean(dim=1)
#         else:
#             # print('else')
#             # print(dense_x[deg_mask,:i].shape)#torch.Size([583, 3, 16])
#             # 找到min最多的那一邻居，删掉(将min最多的那一个邻居特征置为0)
#             min = dense_x[deg_mask, :i].min(dim=1)  # 2
#             # 对min.indices进行处理
#             min_indice = min.indices.mode().values
#             # print(min.indices.mode().values)
#
#             dense_x[deg_mask, :i].index_select(1, min_indice).fill_(0)
#             # 找到max最多的那一邻居，删掉(将max最多的那一个邻居特征置为0)
#
#             max = dense_x[deg_mask, :i].max(dim=1)
#             max_indice = max.indices.mode().values
#             dense_x[deg_mask, :i].index_select(1, max_indice).fill_(0)
#
#             out[deg_mask] = dense_x[deg_mask, :i].mean(dim=1)
#
#     return out

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

