import sys

from deeprobust.graph.defense import GCN,SGC
from deeprobust.graph.targeted_attack import Nettack,FGA,IGAttack,SGAttack,RND
import random
from deeprobust.graph.data import Dpr2Pyg
import scipy.sparse as sp
import json
from utils.utils_2 import Dataset_
import os

path = os.getcwd()

def select_attack_nodes(save_dir=None,dataset=None,dataset_name=None):
    adj = dataset.adj
    idx_test=dataset.idx_test

    degrees = adj.A.sum(0)

    adj1 = adj
    po_edges = []
    print(dataset_name)
    if dataset_name=='acm'or dataset_name=='uai' or dataset_name=='facebookpagepage' or dataset_name=='github':
        for i in range(idx_test.size):
            if degrees[i] > 20:
                po_edges.append(i)
    elif dataset_name=='blogcatalog':
        for i in range(idx_test.size):
            if degrees[i] > 70:
                po_edges.append(i)
    elif dataset_name=='ploblogs':
        for i in range(idx_test.size):
            if degrees[i] > 30:
                po_edges.append(i)
    elif dataset_name=='flickr':
        for i in range(idx_test.size):
            if degrees[i] > 150:
                po_edges.append(i)
    elif dataset_name == 'computers' or dataset_name=='photo':
        for i in range(idx_test.size):
            if degrees[i] > 40:
                po_edges.append(i)
    else:
        for i in range(idx_test.size):
            if degrees[i] > 10:
                po_edges.append(i)

    po_edges = random.sample(po_edges, int(len(po_edges) * 0.5))
    print(os.path.join(save_dir,dataset_name))
    print(len(po_edges))
    with open('{}/{}_attacked_nodes_1.json'.format(save_dir,dataset_name), 'w') as fp:
        json.dump({"attacked_test_nodes": po_edges}, fp)
    return po_edges


def set_surrogate_model(data,attack,device):

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    if attack=='SGAttack':
        surrogate = SGC(nfeat=features.shape[1],
                        nclass=labels.max().item() + 1, K=2,
                        lr=0.01, device=device).to(device)

        pyg_data = Dpr2Pyg(data).pyg_data
        surrogate.fit(pyg_data, verbose=False)
        surrogate.test()

    else:
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate

def set_attack_model(attack,target_node,surrogate,data,j,device):

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    if attack == 'Nettack':
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device).to(
            device)
        model.attack(features, adj, labels, target_node, n_perturbations=j)
    elif attack == 'FGA':

        model = FGA(surrogate, nnodes=adj.shape[0], device=device).to(device)
        model.attack(features, adj, labels, idx_train, target_node, n_perturbations=j)
    elif attack =='IGAttack':
        model = IGAttack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, target_node, j, steps=20)

    elif attack =='SGAttack':
        model = SGAttack(surrogate, attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, j, direct=True)

    elif attack =='RND':
        model = RND()
        model.attack(adj,labels,idx_train,target_node,n_perturbations=j)
    else:
        model = surrogate

    modified_adj = model.modified_adj
    modified_features = model.modified_features
    return modified_adj,modified_features


def generate_targeted_attack(save_dir,targeted_attack_list,dataset_list,device):

    for attack in targeted_attack_list:
        print("--------attack-----:",attack)
        for name in dataset_list:
            print("--------dataset-----:", name)
            data = Dataset_(dataset=name)
            attack_data_save_dir = save_dir+'/'+attack
            if not os.path.exists(attack_data_save_dir):
                os.makedirs(attack_data_save_dir)

            po_edges = select_attack_nodes(attack_data_save_dir,data,name)

            data_adv = data
            for j in range(5):
                print("--------pertubation-----:", j+1)
                for i in range(len(po_edges)):

                    surrogate = set_surrogate_model(data_adv,attack,device)
                    target_node = po_edges[i]
                    modified_adj,modified_features=set_attack_model(attack,target_node,surrogate,data_adv,j+1,device)
                    adj = sp.csr_matrix(modified_adj.A)
                    data_adv.adj = adj

                if not os.path.exists(os.path.join(attack_data_save_dir,name)):
                    os.makedirs(os.path.join(attack_data_save_dir,name))

                sp.save_npz('{}/{}/{}_{}_adj_{}.0'.format(attack_data_save_dir,name,attack,name, j+1), data_adv.adj)


if __name__=='__main__':

    import warnings
    warnings.filterwarnings('ignore')

    from utils.utils_2 import judge_dir,load_param

    param_dict = load_param('attack.yaml')
    device = param_dict['device']
    targeted_attack = param_dict.get('targeted_attack')
    dataset_list = param_dict.get('dataset_list')
    save_dir = param_dict.get('save_dir')

    adv_dataset_save_dir = save_dir
    judge_dir(adv_dataset_save_dir)

    generate_targeted_attack(adv_dataset_save_dir,targeted_attack,dataset_list,device)
