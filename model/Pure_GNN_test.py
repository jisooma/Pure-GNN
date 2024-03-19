# _*_codeing=utf-8_*_
# @Time:2022/11/14  17:10
# @Author:mazhixiu
# @File:Pure_GNN_5.py
# _*_codeing=utf-8_*_
# @Time:2022/10/23  16:45
# @Author:mazhixiu
# @File:Pure_GNN_1.py
# _*_codeing=utf-8_*_
# @Time:2022/10/3  18:00
# @Author:mazhixiu
# @File:GCN_1.py

import sys
sys.path.append('/home/mzx/Pure_GNN/')

import torch
import torch.nn.functional as F

from torch import optim
from copy import deepcopy
from deeprobust.graph import utils as Dprutils


from utils.utils_2 import Dataset_
from torch_geometric.nn import Linear
from torch.nn import Parameter
from model.conv.Pure_GNN_Conv_test import Pure_GNN_Conv
import math

class Pure_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,
                 dropout=0.0,
                 lr=0.01,weight_decay=5e-4,
                 model_param=None,
                 **kwargs):

        super(Pure_GNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.Pure_GNNConv_param = model_param.get('Pure_GNNConv')
        self.Pure_GNN_convs = torch.nn.ModuleList()
        self.Pure_GNN_convs.append(Pure_GNN_Conv(in_channels=in_channels, out_channels=hidden_channels,
                                  layer_param=self.Pure_GNNConv_param.get('layer_1')))
        self.Pure_GNN_convs.append(Pure_GNN_Conv(in_channels=hidden_channels, out_channels=out_channels,
                                  layer_param=self.Pure_GNNConv_param.get('layer_2')))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr

        self.original_x = model_param.get('original_x')
        self.adaptive = model_param.get('adaptive')
        self.n_node = model_param.get('n_node')
        self.bias = Parameter(torch.FloatTensor(1))

        self.lin_src_1 = Linear(in_channels, hidden_channels, bias=False, weight_initializer='glorot')
        self.lin_src_2 = Linear(hidden_channels, out_channels, bias=False, weight_initializer='glorot')

        if self.original_x:
            self.lin_root_x_1 = Linear(in_channels, hidden_channels, bias=False, weight_initializer='glorot')
            self.lin_root_x_2 = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
            if self.adaptive:
                self.beta_1 = Parameter(torch.Tensor(1), requires_grad=True)
                self.beta_2 = Parameter(torch.Tensor(1), requires_grad=True)
            else:
                self.register_buffer('beta', torch.ones(1))

    def reset_parameters(self):
        for conv in self.Pure_GNN_convs:
            conv.reset_parameters()
        if self.original_x and self.adaptive:
            self.beta_1.data.fill_(1)
            self.beta_2.data.fill_(1)

    def forward(self, x, edge_index,):

        # root_x = x
        if self.original_x:
            root_x_1 = self.lin_root_x_1(x)
            root_x_2 = self.lin_root_x_2(x)

        x = self.lin_src_1(x)
        x_g = self.Pure_GNN_convs[0](x, edge_index)
        x_g = F.relu(x_g)
        x_g = F.dropout(x_g, p=self.dropout, training=self.training)
        if self.original_x:
            if self.adaptive:
                attr = F.sigmoid(self.beta_1)
                # attr = self.beta_1
                x_g = attr * x_g + root_x_1 * (1 - attr)
            else:
                x_g = x_g + root_x_1
        else:
            x_g = x

        x_g = self.lin_src_2(x_g)
        x = self.Pure_GNN_convs[1](x_g, edge_index)
        if self.original_x:
            if self.adaptive:
                attr = F.sigmoid(self.beta_2)
                # attr = self.beta_2
                self.output_final = attr * x + root_x_2 * (1 - attr)
            else:
                self.output_final = x + root_x_2

        else:
            self.output_final = x

        x = self.output_final

        return x.log_softmax(dim=-1)


    def fit(self, pyg_data, train_iters=1000, initialize=True, verbose=True, patience=100,device='cpu',**kwargs):

        if initialize:
            self.reset_parameters()
        self.device=device
        self.data = pyg_data

        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose,self.weight_decay,self.lr)


    def train_with_early_stopping(self, train_iters, patience, verbose,weight_decay,lr):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training  ===')
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data.x,self.data.edge_index)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data.x,self.data.edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))


        self.load_state_dict(weights)

    @torch.no_grad()
    def test(self, pyg_data=None):
        self.eval()
        pyg_data =pyg_data
        data = pyg_data.to(self.device) if pyg_data is not None else self.data
        test_mask = data.test_mask
        labels = data.y
        output = self.forward(data.x,data.edge_index)
        self.o = output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = Dprutils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, pyg_data=None):
        self.eval()
        data = pyg_data.data.to(self.device) if pyg_data is not None else self.data
        return self.forward(data.x,data.edge_index)


import time
from utils.utils_2 import judge_dir

def test_1(data,device, model_param, train_param,x_pre=None,att=True):
    save_dir = './attention/'
    judge_dir(save_dir)
    hidden_channels = model_param.get('hidden_channels')
    with_relu = model_param.get('with_relu')
    with_bias = model_param.get('with_bias')

    epochs = train_param.get('epochs')
    lr = train_param.get('lr')
    weight_decay = train_param.get('weight_decay')
    print(device)
    pyg_data = Dpr2Pyg(data).data.to(device)
    model = Pure_GNN(in_channels=pyg_data.x.shape[1],
                hidden_channels=hidden_channels,
                out_channels=data.labels.max().item() + 1,
                with_bias=with_bias,
                with_relu=with_relu,
                lr=float(lr),
                weight_decay=float(weight_decay),
                model_param=model_param,
                ).to(device)
    start = time.perf_counter()
    model.fit(pyg_data, train_iters=epochs, device=device,initialize=True)
    end = time.perf_counter()
    acc = model.test()

    # output = model.o.cpu().numpy()
    # attention = model.Pure_GNN_convs[0].alpha.cpu().numpy()
    # last_layer_hidden_features = model.output_final.cpu().detach().numpy()
    # print(last_layer_hidden_features.shape)
    # first_layer_hidden_features = model.first_layer.cpu().detach().numpy()
    # print('fi')
    # print(first_layer_hidden_features.shape)
    return acc, end - start
    # first_layer_hidden_features
    # return acc,end-start,output

from utils.set_args import *
from deeprobust.graph.data import Dpr2Pyg


def test_Pure_GNN():
    import warnings
    warnings.filterwarnings('ignore')
    from utils.utils_2 import judge_dir, load_param
    from utils.utils_2 import Dataset_
    config_param = load_param('./config_test.yaml')
    dataset_list = config_param.get('dataset_list')
    save_dir = config_param.get('save_dir')

    var_list = []
    acc_list = []
    time_list = []
    path_1 = save_dir + '/acc/'
    path_2 = save_dir + '/var/'
    path_3 = save_dir + '/time/'

    judge_dir(path_1)
    judge_dir(path_2)
    judge_dir(path_3)

    for dataset in dataset_list:
        param_dict = load_param('../config_test/clean/' + dataset + '.yaml')
        device = param_dict['device']
        print(device)
        # global device
        model_param = param_dict.get('model')
        train_param = param_dict.get('train_param')
        data = Dataset_(dataset=dataset, )

        epoch = 5
        sum = 0
        all_time = 0
        var = []
        # output_sum = 0
        for i in range(epoch):
            # 提前计算并保存好
            # acc, times,last = test_1(data, device, model_param, train_param, )
            acc, times = test_1(data, device, model_param, train_param, )
            all_time = all_time + times
            sum = sum + acc
            var.append(acc)


        acc_var = np.std(var)
        mean_acc = sum / epoch
        mean_time = all_time / epoch
        acc_list.append(mean_acc)
        var_list.append(acc_var)
        time_list.append(mean_time)
        print(acc_list)

    np.savetxt("{}/{}_{}.txt".format(save_dir + '/acc', 'clean','Pure_GNN'), np.array(acc_list))
    np.savetxt("{}/{}_{}.txt".format(save_dir + '/var', 'clean', 'Pure_GNN'), np.array(var_list))
    np.savetxt("{}/{}_{}.txt".format(save_dir + '/time', 'clean', 'Pure_GNN'), np.array(time_list))

def test_targeted_attack():
    import warnings
    warnings.filterwarnings('ignore')

    from utils.utils_2 import judge_dir,load_param
    config_param = load_param('./config_test.yaml')
    dataset_list = config_param.get('dataset_list')
    targeted_attack = config_param.get('targeted_attack')
    save_dir = config_param.get('save_dir')

    path_1 = save_dir + '/acc/'
    path_2 = save_dir + '/var/'
    path_3 = save_dir + '/time/'

    judge_dir(path_1)
    judge_dir(path_2)
    judge_dir(path_3)
    test_list = []
    for dataset in dataset_list:
        for attack in targeted_attack:
            # print(attack)

            param_dict = load_param('../config_test/' + attack + '/' + dataset + '.yaml')
            device = param_dict['device']

            model_param = param_dict.get('model')
            train_param = param_dict.get('train_param')

            var_list = []
            acc_list = []
            time_list = []
            for ptb in targeted_ptb_list:
                data = Dataset_(dataset=dataset, attack=attack,ptb=ptb)
                epoch = 5
                sum = 0
                all_time = 0
                var = []
                for i in range(epoch):

                    acc, times = test_1(data, device, model_param, train_param, )

                    all_time = all_time + times
                    sum = sum + acc
                    var.append(acc)

                acc_var = np.std(var)
                mean_acc = sum / epoch
                mean_time = all_time / epoch
                acc_list.append(mean_acc)
                var_list.append(acc_var)
                time_list.append(mean_time)

            test_list.append(acc_list)
            np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc','Pure_GNN',attack,dataset), np.array(acc_list))
            np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var', 'Pure_GNN', attack,dataset,), np.array(var_list))
            np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/time', 'Pure_GNN',attack, dataset,), np.array(time_list))
    # print(test_list[0])
    # print(test_list[1])
    # print(test_list[2])
    # print(test_list[3])
    # print(test_list[4])
    # print(test_list[5])

def test_global_attack():
    import warnings
    warnings.filterwarnings('ignore')
    from utils.utils_2 import judge_dir, load_param

    config_param = load_param('./config_test.yaml')
    dataset_list = config_param.get('dataset_list')
    global_attack = config_param.get('global_attack')
    save_dir = config_param.get('save_dir')
    print(dataset_list)
    print(global_attack)
    path_1 = save_dir + '/acc/'
    path_2 = save_dir + '/var/'
    path_3 = save_dir + '/time/'

    judge_dir(path_1)
    judge_dir(path_2)
    judge_dir(path_3)
    test_list = []
    for dataset in dataset_list:
        for attack in global_attack:
            print(dataset)
            print(attack)
            param_dict = load_param('../config_test/' +attack + '/' + dataset + '.yaml')
            # print(param_dict)
            # print('../config_1/' + attack + '/' + dataset + '.yaml')
            device = param_dict['device']

            model_param = param_dict.get('model')
            train_param = param_dict.get('train_param')

            acc_list = []
            var_list = []
            time_list = []
            for ptb in global_ptb_list:

                if attack=='Metattack':
                   data = Dataset_(dataset=dataset, attack=attack, ptb=ptb,largest_component=False)
                else:
                    data = Dataset_(dataset=dataset, attack=attack, ptb=ptb, )
                print(data)
                epoch = 5
                sum = 0
                all_time = 0
                var = []
                for i in range(epoch):
                    acc, times = test_1(data, device, model_param, train_param, )
                    all_time = all_time + times
                    sum = sum + acc
                    var.append(acc)

                acc_var = np.std(var)
                mean_acc = sum / epoch
                mean_time = all_time / epoch
                acc_list.append(mean_acc)
                var_list.append(acc_var)
                time_list.append(mean_time)
            print(acc_list)
            test_list.append(acc_list)
            np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc', 'Pure_GNN', attack, dataset), np.array(acc_list))
            np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var', 'Pure_GNN', attack, dataset, ), np.array(var_list))
            np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/time', 'Pure_GNN', attack, dataset, ), np.array(time_list))
    # print(test_list[0])
    # print(test_list[1])
    # print(test_list[2])
    # print(test_list[3])
    # print(test_list[4])
    # print(test_list[5])

if __name__=='__main__':
    global_ptb_list =  [0.05, 0.10, 0.15, 0.20, 0.25]
    targeted_ptb_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    # test_Pure_GNN()
    # test_global_attack()
    test_targeted_attack()

