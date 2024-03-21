from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, mul, matmul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from model.components.messagers import MESSAGERS
from model.components.aggregators import AGGREGATORS
from model.components.updators import UPDATORS
from model.components.attentions import ATTENTIONS


from model.components.aggregators import AGGREGATORS
from torch.nn import functional as F


class Pure_GNN_Conv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = False,
                 layer_param=None,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if bias:
            self.bias = Parameter(torch.Tensor(1),requires_grad=True)
        else:
            self.register_parameter('bias_', None)
        self.message_component = dict(layer_param.get('message_component')).get('name')
        print(self.message_component)
        self.att_src = None
        self.att_dst = None

        if self.message_component=='cos_sim':
            self.requires_grad = True
            if self.requires_grad:
                self.beta = Parameter(torch.Tensor(1), requires_grad=True)
            else:
                self.register_buffer('beta', torch.ones(1))

        self.aggregator_component = dict(layer_param.get('aggegator_component'))
        self.aggregators = self.aggregator_component.get('aggregators')
        self.with_attention = self.aggregator_component.get('with_attention')
        self.attention =self.aggregator_component.get('attention')

        if self.with_attention and self.attention != None:
            self.attention = ATTENTIONS[self.attention](out_channels, num_hiddens=16)

        self.num_nodes = layer_param.get('num_nodes')
        self.updator_component = layer_param.get('updator_component')

        if self.updator_component=='initial_x':
            self.adptive = Parameter(torch.Tensor(1), requires_grad=True)

        elif self.updator_component == 'feature_dense_filter':
            self.df = Linear(self.out_channels, 1, bias=False)

        elif self.updator_component == 'feature_sparse_filter':
            self.sf = Parameter(torch.Tensor(1), requires_grad=True)
        else:
            pass


        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

        if self.updator_component == 'feature_dense_filter':
            self.df.reset_parameters()
        if self.updator_component == 'feature_sparse_filter':
            zeros(self.sf)
        if self.message_component=='cos_sim' and self.requires_grad:
            self.beta.data.fill_(1)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        x_norm=None

        self.x_i = x

        if self.message_component == 'cos_sim':
            x_norm = F.normalize(x, p=2, dim=-1)

        out = self.propagate(edge_index,
                             x=x,
                             x_norm=x_norm,
                             edge_weight=edge_weight,
                             size=None
                             )
        return out

    def message(self,
                x_i:Tensor,
                x_j: Tensor,
                edge_weight: OptTensor,
                x_norm_i: Tensor, x_norm_j: Tensor,
                index: Tensor,
                ptr: OptTensor,
                size_i=Optional[int],
                ) -> Tensor:

       if self.message_component=='const':
           msg = MESSAGERS[self.message_component](x_j=x_j,)

       elif self.message_component == 'cos_sim':
           msg, alpha = MESSAGERS['cos_sim_1'](x_j=x_j, x_norm_i=x_norm_i, x_norm_j=x_norm_j,
                                                          index=index, ptr=ptr, size_i=size_i, )

       else:
           msg = MESSAGERS[self.message_component](x_j=x_j, )

       return msg

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:

        if self.aggregators =='None' :
            return AGGREGATORS['sum'](inputs, index, dim_size)

        outs = [AGGREGATORS[aggr](inputs, index, dim_size) for aggr in self.aggregators]
        out_stack = torch.stack(outs, dim=1)
        if self.with_attention:
            result, aggr_max,alpha = self.attention(out_stack, out_stack, out_stack)  #
            self.alpha = alpha
        else:
            result = out_stack.sum(1)

        return result

    def update(self, inputs: Tensor) -> Tensor:

        if self.updator_component=='initial_x':
            upd = UPDATORS[self.updator_component](inputs,self.adptive,self.x_i)
        elif self.updator_component=='feature_dense_filter':
            upd = UPDATORS[self.updator_component](inputs, self.x_i,self.df)
        elif self.updator_component=='feature_sparse_filter':
            upd = UPDATORS[self.updator_component](inputs, self.x_i,self.sf)
        else:
            upd = UPDATORS['const'](inputs)

        return upd


    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        if self.aggregators == None:
            return matmul(adj_t, x, reduce=self.aggr)

        out_list = []
        for aggr in self.aggregators:
            out = matmul(adj_t, x, reduce=aggr)
            out_list.append(out)
        out_stack = torch.stack(out_list, dim=1)

        if self.with_attention:
            result, aggr_max = eval(
                self.attention(out_stack, out_stack, out_stack))
        else:
            result = out_stack.sum(1)
        return result

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



