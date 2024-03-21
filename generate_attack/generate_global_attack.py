import torch, gc


from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack,MinMax,DICE,Random,Metattack
from deeprobust.graph.utils import preprocess
import scipy.sparse as sp
import os
from utils.utils_2 import Dataset_
import numpy as np

def set_surrogate_model(data):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate


def generate_random_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir,'Random',dataset)

    judge_dir(adj_dir)
    data = Dataset_(dataset=dataset)
    adj, features, labels = data.adj, data.features, data.labels

    for i in range(5):
        model = Random(attack_features=False,attack_structure=True)
        ptb_rate = 0.05 * (i + 1)
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(adj,n_perturbations=n_perturbations)
        modified_adj = model.modified_adj

        save_random_dir ='{}/Random_{}_adj_{}'.format(adj_dir,dataset,round(ptb_rate,3))
        sp.save_npz(save_random_dir,sp.csr_matrix(modified_adj))

def generate_dice_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir,'Dice',dataset)
    judge_dir(adj_dir)

    data = Dataset_(dataset=dataset)

    adj, features, labels = data.adj, data.features, data.labels
    model = DICE()
    for i in range(5):
        ptb_rate = 0.05+(i+1)
        n_perturbations = int(ptb_rate* (adj.sum() // 2))

        model.attack(adj,labels,n_perturbations=n_perturbations)
        modified_adj = model.modified_adj

        save_dice_dir = '{}/Dice_{}_adj_{}'.format(adj_dir,dataset, round((i + 1) * 0.05, 3))
        sp.save_npz(save_dice_dir,sp.csr_matrix(modified_adj))

def generate_PGDAttack_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir, 'PGDAttack', dataset)
    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)

    for i in range(5):
        data = Dataset_(dataset=dataset)
        print(data.features.shape)
        adj, features, labels = data.adj, data.features, data.labels
        adj,features,labels = preprocess(adj,features,labels,preprocess_adj=False)
        idx_train,idx_val,idx_test = data.idx_train,data.idx_val,data.idx_test
        victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                           nhid=16, dropout=0.5, weight_decay=5e-4, device=device).to(device)
        victim_model.fit(features, adj, labels, idx_train)
        model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device,
                          attack_structure=True,attack_features=False).to(device)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        ptb_rate = 0.05 * (i + 1)
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(features, adj, labels, idx_train,n_perturbations=n_perturbations)
        modified_adj = model.modified_adj.cpu().numpy()
        save_PGDAttack_dir = '{}/PGDAttack_{}_adj_{}'.format(adj_dir, dataset, round(ptb_rate, 3))
        sp.save_npz(save_PGDAttack_dir, sp.csr_matrix(modified_adj))


def generate_MinMax_adj(dataset,save_dir):
    adj_dir = os.path.join(save_dir, 'MinMax', dataset)
    if not os.path.exists(adj_dir):
        os.makedirs(adj_dir)

    for i in range(5):
        data = Dataset_(dataset=dataset)
        adj, features, labels = data.adj, data.features, data.labels

        adj, features, labels = preprocess(adj,features,labels,preprocess_adj=False)
        idx_train,idx_val,idx_test = data.idx_train,data.idx_val,data.idx_test
        victim_model = set_surrogate_model(data)

        model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device,
                       attack_features=False,attack_structure=True)
        model = model.to(device)
        ptb_rate = 0.05
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj.cpu().numpy()
        save_MinMax_dir = '{}/MinMax_{}_adj_{}'.format(adj_dir, dataset, round(ptb_rate, 3))

        sp.save_npz(save_MinMax_dir, sp.csr_matrix(modified_adj))



def generate_Metattack_adj(dataset,save_dir):

    adj_dir = os.path.join(save_dir, 'Metattack', dataset)
    judge_dir(adj_dir)
    data = Dataset_(dataset=dataset)
    adj, features, labels = data.adj, data.features, data.labels
    features = sp.csr_matrix(features)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    surrogate = set_surrogate_model(data)

    print(device)
    for i in range(5):

        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device,
                          attack_features=False, attack_structure=True)
        model = model.to(device)
        ptb_rate = 0.05*(i+1)
        n_perturbations = int(ptb_rate * (adj.sum() // 2))
        model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)
        modified_adj = model.modified_adj
        modified_adj_np = sp.csr_matrix(modified_adj.cpu().numpy())

        sp.save_npz('{}/{}/{}/{}_{}_adj_{}'.format(save_dir, 'Metattack',dataset, 'Metattack',dataset,round((i + 1) * 0.05, 3)),
                     modified_adj_np)


if __name__=='__main__':

    import warnings
    warnings.filterwarnings('ignore')

    from utils.utils_2 import judge_dir, load_param

    param_dict = load_param('attack.yaml')
    device = param_dict['device']
    dataset_list = param_dict.get('dataset_list')
    save_dir = param_dict.get('save_dir')

    judge_dir(save_dir)
    global_attack = list(param_dict.get('global_attack'))
    for attack in global_attack:
        for dataset in dataset_list:
            if attack =='Dice':
                generate_dice_adj(dataset,save_dir)
            elif attack =='Random':
                generate_random_adj(dataset,save_dir)
            elif attack=='PGDAttack':
                generate_PGDAttack_adj(dataset,save_dir)
            elif attack=='MinMax':
                generate_MinMax_adj(dataset,save_dir)
            elif attack=='Metattack':
                generate_Metattack_adj(dataset,save_dir)
            else:
                assert ('error')
