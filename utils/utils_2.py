import sys

import numpy as np
import torch
from deeprobust.graph.data import Dpr2Pyg,Pyg2Dpr
from deeprobust.graph.data.dataset import get_train_val_test
from utils.set_args import *

from scipy import sparse as sp

from torch_geometric.datasets import CitationFull, Coauthor,Amazon # EmailEUCore,LastFMAsia,DeezerEurope,Actor,Airports,FacebookPagePage,GitHub,WikiCS,Twitch

import torch_geometric.utils as pygUtils
import yaml
import os

def load_param(filename):
    stream = open(filename, 'r',encoding='utf-8')
    docs = yaml.load_all(stream.read(),Loader = yaml.FullLoader)
    param_dict = dict()

    for doc in docs:
        for k, v in doc.items():
            param_dict[k] = v

    return param_dict

from torch_scatter import scatter
def load_subgraph_1_hop_features(data,dataset,attack=None,ptb=None,save_dir='./subgraph/'):

    if attack==None:
        path = save_dir
        judge_dir(save_dir)
    else:
        path = save_dir+attack+'/'+ptb
        judge_dir(path)

    nodes_list = np.arange(data.features.shape[0])
    pyg_data = Dpr2Pyg(data).data
    features = data.features

    if not os.path.exists(path+'/subgraph_features_{}.txt'.format(dataset)):

        subset_list = []
        for node in nodes_list:

            subset, edge_index, mapping, edge_mask = pygUtils.k_hop_subgraph(int(node), 1, pyg_data.edge_index)

            neighbor_feature_set = data.features[subset]

            subset_x = drop_perturb_edges(threshold=0.01,features=neighbor_feature_set,
                                               nodes_list=subset,center_node=mapping)


            subset_list.append(subset_x)

        edge_index_pre = merge_subgraph(subset_list)

        edge_index_fea = merge_subgraph_features(edge_index_pre,features)

        np.savetxt(path+'/subgraph_features_{}.txt'.format(dataset),edge_index_fea)
        np.savetxt(path+'/subgraph_adj_{}.txt'.format(dataset),edge_index_pre)

        x_pre = scatter(torch.FloatTensor(edge_index_fea), index=torch.LongTensor(edge_index_pre), dim=-2, reduce='sum')
        np.savetxt(path + '/subgraph_x_pre_{}.txt'.format(dataset), x_pre)

        return x_pre
    else:
        print('loading subgraph')
        x_pre = np.loadtxt(path + '/subgraph_x_pre_{}.txt'.format(dataset))
        return x_pre


def merge_subgraph(subset_list):
    edge_index = []
    for i in subset_list:
        edge_index.extend(i)
    return edge_index

def merge_subgraph_features(edge_index_pre,features):

    features_j = []
    for i in edge_index_pre:
        if type(features) is not np.ndarray:
            features_j.append(features[i].todense())
        else:
            features_j.append(features[i])

    if type(features) is not np.ndarray:
        return np.array(features_j).squeeze(axis=1)
    else:
        return np.array(features_j)

def drop_perturb_edges(threshold,features,nodes_list,center_node,binary_features=True):

    if type(features) is not np.ndarray:
        features = features.toarray()
    center_node_feature = features[center_node]

    all_nodes_list = nodes_list.tolist()
    nodes_list = nodes_list.tolist()

    for node in range(len(nodes_list)):
        if binary_features:
            J = jaccard_similarity(center_node_feature,features[node])
            if J < threshold:
                nodes_list.remove(all_nodes_list[node])
        else:
            C = cosine_similarity(center_node_feature,features[node])
            if C < threshold:
                nodes_list.remove(nodes_list[node])
    return nodes_list

def jaccard_similarity(a,b):

    intersection = np.multiply(a,b)
    intersection = np.count_nonzero(intersection)
    a_no = np.count_nonzero(a)
    b_no = np.count_nonzero(b)
    if a_no+b_no==intersection:
        J = 0
    else:
        J = intersection*1.0/(a_no+b_no-intersection)
    return J

def cosine_similarity(a,b):
    inner_product = (a*b).sum()
    C = inner_product/(np.sqrt(np.square(a).sum())*np.sqrt(np.square(b).sum())+1e-10)
    return C


def judge_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        os.makedirs(path)


def accuracy_with_attack_nodes(output, labels):


    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()

    correct = correct.sum()
    return correct / len(labels)



def accuracy_with_error_classify_nodes(output, labels,test_mask):

    print(len(test_mask.nonzero()))

    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()

    error_class = torch.nonzero(correct == 0)
    error_class_mask = torch.zeros(len(test_mask))
    error_class_mask[error_class] = 1

    correct = correct.sum()
    return correct / len(labels),error_class,error_class_mask


def dir_(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

experiment_attention_dir = './attention_distribution/'

def save_aggr_distribution(model,dataset,layer,aggregators,attention,aggr_max,attack=None,ptb=None):

    path = experiment_attention_dir+model
    if not os.path.exists(path):
        os.makedirs(path)
    if attack == None:
        np.savetxt("{}/{}_{}_{}_{}.txt".format(path, model,dataset, layer,aggregators,attention), np.array(aggr_max))
    else:
        np.savetxt("{}/{}_{}_{}_{}_{}_{}.txt".format(path, model,dataset, layer,aggregators,attention,
                                                     attack,ptb), np.array(aggr_max))


experiment_output_dir = './output_representation'
def save_output(output,d_name,attention,aggregators,attack=None,ptb=None):

    if not os.path.exists(experiment_output_dir):
        os.makedirs(experiment_output_dir)
    if attack==None:
        np.save("{}/{}_{}_{}".format(experiment_output_dir, d_name, attention, aggregators),output)
    else:
        np.save("{}/{}_{}_{}_{}_{}".format(experiment_output_dir, d_name, attention, aggregators,attack,ptb), output)


def from_scipy_sparse_matrix(A):

    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


class Dataset_Attack():
    def __init__(self,dataset):
        self.dataset = dataset
        self.adj = None
        self.features = None
        self.labels= None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2},labels={3},idx_train={4},idx_val={5},idx_test={6})'.format(
            'Dataset_Attack:'+self.dataset, self.adj.shape, self.features.shape,
            self.labels.shape,self.idx_train.shape,self.idx_val.shape,self.idx_test.shape)

from utils.set_args import *
from deeprobust.graph.data import Dataset
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph

class Dataset_():
    print(clean_dataset_dir)
    def __init__(self,clean_dir=clean_dataset_dir,
                 dataset='cora',
                 attack=None,
                 attack_dir=attack_data_dir,
                 ptb=None,
                 largest_component=True,
                 is_cut=True,
                 num_nodes_dict=cut_n_dict_2):

        self.clean_dir = clean_dir
        self.dataset = dataset.lower()

        self.attack = attack
        self.attack_dir = attack_dir
        self.ptb = ptb
        if self.attack=='Metattack' and self.dataset=='photo':
            largest_component = False

        self.largest_component = largest_component

        self.is_cut = is_cut
        self.num_nodes_dict = num_nodes_dict

        assert self.dataset in ['acm', 'cora', 'citeseer', 'cora_ml',  'pubmed','photo',
                                'polblogs','blogcatalog', 'uai', 'flickr','physics','dblp','computers',
                                'cs','actor','meta_photo'
                                ], \
                'Currently only support  pubmed, acm, cora, citeseer, cora_ml,' # ' polblogs, blogcatalog, flickr'

        if attack==None:
            self.adj, self.features, self.labels,self.idx_train,self.idx_val,\
                                        self.idx_test = self.load_clean_data(self.clean_dir,
                                                                             self.dataset)
        else:
            if ptb==None:
                assert 'please specify attack ptb'
            self.adj, self.features, self.labels,self.idx_train,self.idx_val,\
                                        self.idx_test = self.load_attack_data(self.dataset,
                                                                              self.attack,
                                                                              self.attack_dir,
                                                                              self.ptb)
    def is_largest_component(self,pyg_data):
        nx_data = to_networkx(pyg_data, to_undirected=True)
        flag = nx.is_connected(nx_data)
        return nx_data,flag

    def load_largest_component(self,nx_data,data):

        largest_cc = max(nx.connected_components(nx_data), key=len)
        largest_cc = list(largest_cc)
        num_nodes = len(largest_cc)
        features = np.array(data.x[largest_cc])
        row, col = np.nonzero(features)
        values = features[row, col]
        features = sp.csr_matrix((values, (row, col)), shape=features.shape)
        labels = np.array(data.y[largest_cc])

        subset = torch.LongTensor(largest_cc)
        edge_index, edge_mask = subgraph(subset=subset, edge_index=data.edge_index,relabel_nodes=True)
        idx_train, idx_val, idx_test = get_train_val_test(
            len(labels), val_size=0.1, test_size=0.8, stratify=labels, seed=15)
        adj = pygUtils.to_scipy_sparse_matrix(edge_index).tocsr()

        return adj, features,labels, idx_train, idx_val, idx_test

    def load_clean_data(self,clean_dataset_dir,name):

        if name =='dblp':
            pyg_data = CitationFull(root=clean_dataset_dir, name=name)
        elif name == 'cs' or name=='physics':
            pyg_data = Coauthor(root=clean_dataset_dir,name=name)
            print(pyg_data)
        elif name =='computers' or name=='photo':
            pyg_data = Amazon(root=clean_dataset_dir,name=name)
        elif name =='cora' or name=='cora_ml'or name=='citeseer'or name=='polblogs'or name=='pubmed'\
                or name =='acm' or name=='flicker' or name=='uai' or name=='flickr' or name=='blogcatalog':
            dpr_data = Dataset(root=clean_dataset_dir,name=name,seed=15)
            pyg_data = Dpr2Pyg(dpr_data)

        elif name=='meta_photo':
            name = 'photo'
            pyg_data = Amazon(root=clean_dataset_dir,name=name)
            print(pyg_data)
            name = 'meta_photo'
        else:
            assert name + "is not be supported！"

        dpr_data = Pyg2Dpr(pyg_data.data)
        print('loading '+ self.dataset+' dataset.....')
        if self.is_cut:
            if self.attack=='Metattack' and self.dataset=='photo':
                name = 'meta_photo'

            if name == 'meta_photo':
                self.largest_component=False

            adj, features, labels, idx_train, idx_val, idx_test = self.cutDataSet(dpr_data, self.num_nodes_dict[name])

            row, col = np.nonzero(features)
            values = features[row, col]
            features = sp.csr_matrix((values, (row, col)), shape=features.shape)

            if self.largest_component:
                data = self.toPygData(adj, features, labels, idx_train, idx_val, idx_test)
                nx_data, flag = self.is_largest_component(data)
                if not flag:
                    adj, features, labels, idx_train, idx_val, idx_test \
                        = self.load_largest_component(nx_data,data)

            row, col = np.diag_indices_from(adj)
            adj[row, col] = 1

            return adj, features, labels, idx_train, idx_val, idx_test

        else:
            self.is_largest_component(pyg_data)
            return dpr_data.adj,dpr_data.features,dpr_data.idx_train,dpr_data.idx_val,dpr_data.idx_test


    def index_to_mask(self,index, size):
        mask = torch.zeros((size,), dtype=torch.bool)
        mask[index] = 1
        return mask

    def toPygData(self,adj, features, labels, idx_train, idx_val, idx_test):
        # Dpr2Pyg
        edge_index = torch.LongTensor(adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(features):
            x = torch.FloatTensor(features.todense()).float()
        else:
            x = torch.FloatTensor(features).float()
        y = torch.LongTensor(labels)

        data = Data(x=x, edge_index=edge_index, y=y)
        train_mask = self.index_to_mask(idx_train, size=y.size(0))
        val_mask = self.index_to_mask(idx_val, size=y.size(0))
        test_mask = self.index_to_mask(idx_test, size=y.size(0))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def load_attack_data(self,name, attack, attack_dir,ptb):
        adj, features, labels, idx_train, idx_val, idx_test = self.load_clean_data(clean_dataset_dir, name)

        if attack =='Metattack' or attack=='Nettack' or  attack=='SGAttack' or attack=='Dice':
            data_attack = Dataset_Attack(name)
            data_attack.adj = adj
            data_attack.features = features
            data_attack.labels = labels
            data_attack.idx_train = idx_train
            data_attack.idx_val = idx_val
            data_attack.idx_test = idx_test
            adj, features, labels, idx_train, idx_val, idx_test = self.cutDataSet(data_attack,self.num_nodes_dict[name])

        features = features
        if name=='meta_photo':
            name='photo'
        path = attack_dir + '/' + attack + '/' + name
        perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.npz'.format(path, attack, name, float(ptb)))
        adj = perturbed_adj
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)

        row, col = np.diag_indices_from(adj)
        adj[row, col] = 1

        return adj,features, labels, idx_train, idx_val, idx_test

    def cutDataSet(self,data, k):
        """

        :param data:
        :return:
        """
        print(k)
        num_node = len(data.labels)
        k = min(k, num_node)

        keep_nodes = np.array([i for i in range(k)])
        features = data.features[0:k]
        labels = data.labels[0:k]

        edges = data.adj.nonzero()
        e0 = np.array(edges[0])
        e1 = np.array(edges[1])
        edge_index_array = np.array(list(zip(e0, e1)))
        edge_index = torch.from_numpy(edge_index_array.T)
        nodes = torch.LongTensor(keep_nodes)

        edge_index = pygUtils.subgraph(nodes, torch.LongTensor(edge_index.long()))[0]

        idx_train, idx_val, idx_test = get_train_val_test(
            len(labels), val_size=0.1, test_size=0.8, stratify=labels, seed=15)

        features = features
        labels = labels
        adj = pygUtils.to_scipy_sparse_matrix(edge_index).tocsr()
        idx_train = idx_train
        idx_val = idx_val
        idx_test = idx_test

        return adj,features,labels,idx_train,idx_val,idx_test

    def heterophily_handle(self,pyg_data,):
        n = pyg_data.num_nodes
        self.idx_train = self.mask_to_index(pyg_data.train_mask,n)
        self.idx_val = self.mask_to_index(pyg_data.val_mask, n)
        self.idx_test = self.mask_to_index(pyg_data.test_mask, n)

    def mask_to_index(self,index, size):
        all_idx = np.arange(size)
        return all_idx[index]

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2},labels={3},idx_train={4},idx_val={5},idx_test={6})'.format(
            'Dataset_PureGNN:'+self.dataset, self.adj.shape, self.features.shape,
            self.labels.shape,self.idx_train.shape,self.idx_val.shape,self.idx_test.shape)


import json
def load_attack_nodes(attack,dataset):
    with open(attack_data_dir+attack+'/'+dataset+'_attacked_nodes_1.json', "r", encoding="utf-8") as f1:
        attack_nodes_json = json.load(f1)
        attack_nodes = attack_nodes_json['attacked_test_nodes']
        attack_nodes_sort = np.sort(attack_nodes)
        return attack_nodes_sort


def test_adj(adj_clean,adj_attack):

    adj_add = adj_attack-adj_clean
    adj_add_ = (adj_add==1).nonzero()

    # delete
    adj_delete = adj_clean - adj_attack
    adj_delete_ = (adj_delete == 1).nonzero()
    node_list_1 = np.unique(adj_add_[0])
    node_list_2 = np.unique(adj_add_[1])
    node_list_3 = np.unique(adj_delete_[0])
    node_list_4 = np.unique(adj_delete_[1])

    node_list = np.concatenate((node_list_1,node_list_2,node_list_3,node_list_4))
    attacked_node = np.unique(node_list)
    print(len(attacked_node))

    return adj_delete_,adj_add_,attacked_node


def all_attack_nodes():
    save_dir = './attack_nodes/'
    judge_dir(save_dir)
    for dataset in ['photo']:
        clean_data = Dataset_(dataset=dataset,largest_component=False)
        print(clean_data)
        clean_adj  = clean_data.adj
        for attack in ['Metattack',]:
            perturbed_data = Dataset_(dataset=dataset, attack=attack, ptb='0.25', )
            perturbed_adj = perturbed_data.adj
            adj_delete_,adj_add_,attacked_node = test_adj(adj_clean=clean_adj,adj_attack=perturbed_adj)
            if dataset=='meta_photo':
                dataset='photo'
            np.savetxt('{}/{}_{}_0.25.txt'.format(save_dir,attack,dataset),np.array(attacked_node))

        for attack in ['SGAttack','Nettack']:
            perturbed_data = Dataset_(dataset=dataset, attack=attack, ptb='5.0', largest_component=True)
            perturbed_adj = perturbed_data.adj
            adj_delete_,adj_add_,attacked_node = test_adj(adj_clean=clean_adj,adj_attack=perturbed_adj)
            np.savetxt('{}/{}_{}_5.0.txt'.format(save_dir, attack, dataset), np.array(attacked_node))

import matplotlib.pyplot as plt
def degree_distribution(degree_sequence,attack=None,dataset=None):
    import datetime
    now = datetime.datetime.now()
    path_1 ='./deg_distribution_2023_5_23/'
    judge_dir(path_1)

    import numpy as np
    fig,ax = plt.subplots()
    ax.bar(*np.unique(degree_sequence,return_counts=True))
    if attack==None:
        ax.set_title(dataset+'  Degree Distribution ')
    else:
        ax.set_title(dataset + '  Degree Distribution in ' + attack)
    ax.set_xlabel('Degree')
    ax.set_ylabel('#of Nodes')

    if attack==None:
        plt.savefig(os.path.join(path_1,dataset) + '.jpg',
                    dpi=80, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(path_1, attack + '_' + dataset) + '.jpg',
                    dpi=80, bbox_inches='tight')
    plt.show()

def degree():
    x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
    y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
    plt.scatter(x, y, color='hotpink')


def count_degree():
    save_dir = './attack_nodes/'
    judge_dir(save_dir)

    for dataset in ['acm', 'cora_ml', 'citeseer']:
        clean_data = Dataset_(dataset=dataset, largest_component=True)
        clean_adj = clean_data.adj

        for attack in ['Dice', 'Random', 'Metattack']:
            perturbed_data = Dataset_(dataset=dataset, attack=attack, ptb='0.25', largest_component=True)
            perturbed_adj = perturbed_data.adj
            attacked_node = np.loadtxt('{}/{}_{}_0.25'.format(save_dir, attack, dataset))

            c_d_list = []
            p_d_list = []
            d_list = []
            for node in attacked_node:
                c_d = clean_adj[node].sum()

                p_d = perturbed_adj[node].sum()
                d = p_d - c_d

                c_d_list.append(int(c_d))
                p_d_list.append(int(p_d))
                d_list.append(int(d))


            with open('{}/{}_{}_attacked_nodes_degree.json'.format(save_dir, dataset,attack), 'w') as fp:
                degree_dict = dict()
                degree_dict["clean_degree"] = c_d_list
                degree_dict["attack_degree"] = p_d_list
                degree_dict["sub_degree"] = d_list

                json.dump(degree_dict, fp)


        for attack in ['SGAttack', 'Nettack']:
            perturbed_data = Dataset_(dataset=dataset, attack=attack, ptb='5.0', largest_component=True)
            perturbed_adj = perturbed_data.adj
            attacked_node = np.loadtxt('{}/{}_{}_5.0'.format(save_dir, attack, dataset))

            c_d_list = []
            p_d_list = []
            d_list = []
            for node in attacked_node:
                c_d = clean_adj[node].sum()
                p_d = perturbed_adj[node].sum()
                d = p_d - c_d

                c_d_list.append(int(c_d))
                p_d_list.append(int(p_d))
                d_list.append(int(d))

                degree_distribution(sorted(c_d_list), attack, dataset)

            with open('{}/{}_{}_attacked_nodes_degree.json'.format(save_dir, dataset,attack), 'w') as fp:
                degree_dict = dict()
                degree_dict["clean_degree"] = c_d_list
                degree_dict["attack_degree"] = p_d_list
                degree_dict["sub_degree"] = d_list

                json.dump(degree_dict, fp)



def load_degree():
    save_dir = './attack_nodes/'
    judge_dir(save_dir)
    # 'acm',
    for dataset in ['acm']:#  'cora_ml', 'citeseer'
        clean_data = Dataset_(dataset=dataset, largest_component=True)
        clean_adj = clean_data.adj

        for attack in ['Dice', 'Random', 'Metattack']:
            perturbed_data = Dataset_(dataset=dataset, attack=attack, ptb='0.25', largest_component=True)
            perturbed_adj = perturbed_data.adj
            with open('{}/{}_{}_attacked_nodes_degree.json'.format(save_dir, dataset,attack), 'r') as fp:
                data_str = fp.read()
                degree = json.loads(data_str)
                sub_degree = degree['sub_degree']
                clean_degree = degree['clean_degree']
                attack_degree = degree['attack_degree']
                degree_distribution(sorted(sub_degree), attack, dataset+'_sub')
                degree_distribution(sorted(clean_degree), attack, dataset+'_clean')
                degree_distribution(sorted(attack_degree), attack, dataset+'_attack')

        for attack in ['SGAttack', 'Nettack']:
            perturbed_data = Dataset_(dataset=dataset, attack=attack, ptb='5.0', largest_component=True)
            perturbed_adj = perturbed_data.adj
            with open('{}/{}_{}_attacked_nodes_degree.json'.format(save_dir, dataset,attack), 'r') as fp:
                data_str = fp.read()
                degree = json.loads(data_str)

                sub_degree = degree['sub_degree']
                clean_degree = degree['clean_degree']
                attack_degree = degree['attack_degree']
                degree_distribution(sorted(sub_degree), attack, dataset+'_sub')
                degree_distribution(sorted(clean_degree), attack, dataset+'_clean')
                degree_distribution(sorted(attack_degree), attack, dataset+'_attack')


def degree_statics():
    for dataset in [ 'acm','citeseer','cora_ml','uai','pubmed']:
        # pass
        clean_data = Dataset_(dataset=dataset,)
        clean_adj = clean_data.adj
        print(len(clean_adj.nonzero()[0]))
        print(len(clean_adj.nonzero()[0])/2)
        degree = np.ravel(clean_adj.sum(axis=1))
        print(np.sum(degree))
        print(np.min(degree))
        print(np.median(degree))
        print(np.max(degree))
        print(np.sum(degree)/len(degree))


def load_attack_nodes_5_0(attack,dataset):
    dir = '/'
    attack_node = None
    if attack in ['Dice','Random','Metattack']:
        attacked_node = np.loadtxt('{}/{}_{}_0.25.txt'.format(dir, attack, dataset))
    elif attack in ['SGAttack','Nettack']:
        attacked_node = np.loadtxt('{}/{}_{}_5.0.txt'.format(dir, attack, dataset))
    else:
        print('error')

    return attacked_node.astype(int)


def load_attack_nodes_degree_5_0(attack,dataset):
    dir = '/'
    with open('{}/{}_{}_attacked_nodes_degree.json'.format(dir, dataset, attack), 'r') as fp:
        data_str = fp.read()
        degree = json.loads(data_str)

        sub_degree = degree['sub_degree']
        clean_degree = degree['clean_degree']
        attack_degree = degree['attack_degree']

    return attack_degree

if __name__=='__main__':

    import warnings
    warnings.filterwarnings('ignore')

    dataset_list = ['acm','citeseer','cora_ml','uai', 'pubmed','photo'] #
    for dataset in dataset_list:
        data = Dataset_(dataset=dataset)
        print(len(data.adj.nonzero()[0]))
