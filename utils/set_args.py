

# PROJECT_DIR = '/home/mzx/CodeSet/Pure_GNN_test/'
#
#
# import sys
# sys.path.append(PROJECT_DIR) # 设置项目根路径


import torch
import numpy as np


"""
# 数据集列表
#[,,'Cora'
dataset_list = ['cora','cora_ml','citeseer', 'pubmed']
# 攻击列表:'SGAttack','IGAttack',
attack_list = ['Nettack','FGA','Random','Dice','PGDAttack','MinMax','Metattack']
# 防御列表
defense_list = ['GCN','GAT','RGCN','GCNSVD','GCNJaccard_0106', 'GNNGuard','ProGCN','SimpGCN','MedianGCN','ElasticGCN']
# 定向攻击列表
# ,
targeted_attack_list = ['RND','Nettack','SGAttack'] # 'IGAttack','FGA',
# global_attack_list = ['Random','Dice','PGDAttack','MinMax']
# 全局攻击列表
global_attack_list = ['Random','Dice','Metattack'] #'PGDAttack','MinMax',

# 干净数据集保存路径
clean_dataset_dir=path+'/clean_data/'
print(clean_dataset_dir)
# 对抗样本保存路径
adv_dataset_dir=path+'/attack_data_925'
# 对比实验数据保存路径
experiment_data_dir = path+'/experiment_data'
# 对比试验图表保存路径
experiment_figure_dir = path + '/experiment_figure'
# 模型保存路径
model_save_dir = path+'/model_save'


# 防御列表
defense_dict = {
                'GCN':'GCN_',
                'GAT':'GAT_',
                'RGCN':'R_GCN',
                # 'GCNSVD':'GCN_SVD',
                # 'GCNJaccard_0106':'GCN_Jaccard',
                # 'ProGCN':'Pro_GCN',
                'GNNGuard':'GNN_Guard',
                'MedianGCN': 'Median_GCN',#
                'ElasticGCN': 'Elastic_GCN',
                'SimpGCN':'Simp_GCN',
                }
                

#
# knn_n = {
#         'cora':60,
#         'cora_ml':60,
#         'citeseer.yaml':60,
#         'polblogs':60,
#         'pubmed':60,
#
#         'acm':3,
#         'uai':19,
#         'blogcatalog': 6,
#         'flickr':9,
#
#         'cs':15,
#         'physics':5,
#         'dblp':4,
#         'computers':10,
#         'photo':8,
# }
# dataset_list = ['cora', 'citeseer',  'cora_ml', 'pubmed']
#
# # path_acc = './baseline_defense/'
#
# attack_list_1 = ['Dice','Metattack']
# attack_list_2 = ['Nettack','SGAttack']

"""
global_ptb_list = ['0.05','0.10','0.15', '0.20', '0.25']
targeted_ptb_list = ['1.0','2.0','3.0','4.0','5.0']
# path = PROJECT_DIR

# cuda = torch.cuda.is_available()
# device = torch.device('cuda:1'if torch.cuda.is_available() else 'cpu')
#
# seed=15
# np.random.seed(seed)
# torch.manual_seed(seed)
# if cuda:
#     torch.cuda.manual_seed(seed)

# 干净数据集存放目录
# clean_dataset_dir ='D:\\TASKS\\graph_defense\\DATA\\clean_data\\'
# 攻击数据集存放目录
# attack_data_dir ='D:\\TASKS\\graph_defense\\DATA\\attack_data\\'


# 干净数据集存放目录
clean_dataset_dir ='/home/mzx/TASKS/DATA/DATA/clean_data/'
# # 攻击数据集存放目录
attack_data_dir ='/home/mzx/TASKS/DATA/DATA/attack_data/'
#
# # clean_dataset_dir ='/home/mzx/Code/MAGNET_1/clean_data/'
# # 攻击数据集存放目录
# # attack_data_dir ='/home/mzx/Code/MAGNET_v3/attack_data_1204/'
# attack_data_dir ='/home/mzx/DATA/attack_data_423/attack_data/'
# # attack_data_dir ='/home/mzx/DATA/attack_data_1/'

# attack_data_dir ='/home/mzx/DATA/attack_data/'
# cut_n_dict={
#         'cora':2485,
#         'cora_ml':2810,
#         'citeseer':2110,
#         'polblogs':1222,
#         # 'pubmed':15000,
#         'pubmed':19717,
#
#         'acm':2000,# 2500 如果划分为2000，少一个类
#         'uai':1500,
#         'blogcatalog': 1000,
#         'flickr':500,
#
#         'cs':8000,
#         'physics':8000,
#         'dblp':8000,
#         'computers':5000,
#         'photo':4000,
# }
#
# cut_n_dict_1={
#         'cora':2485,
#         'cora_ml':2810,
#         'citeseer':2110,
#         'polblogs':1222,
#         # 'pubmed':15000,
#         'pubmed':19717,
#
#         'acm':2500,# 存在多个连通组件
#         'uai':1500,# 存在多个连通组件
#         'blogcatalog': 1000,
#         'flickr':500,
#
#         'cs':8000,
#         'physics':8000,
#         'dblp':8000,
#         'computers':5000,
#         'photo':4000,
# }

cut_n_dict_2={
        'cora':2485,
        'cora_ml':2810,

        'citeseer':2110,
        'polblogs':1222,
        'pubmed':19717,

        'acm':3025,# 修改

        # 'blogcatalog': 1000,
        # 'uai':2000,#  针对Meattack
        'uai':3067,# 修改
        # 'flickr':500,

        # 'cs':18333,
        'cs':12000,
        'physics':8000,
        # 'dblp':8000,
        # 'computers':5000,# 切割
        'computers': 13752,
        'meta_photo':4000,# 针对Meattack

        'photo':7650,

        # 'lastfmasia': 7624,
        # 'facebookpagepage': 12000,
        # 'deezereurope': 18000,
        # 'github': 15000,
        #
        # 'usa': 1190,
        # 'brazil': 131,
        # 'europe': 399,
        #
        # 'DE': 9498,
        # "EN": 7126,
        # "ES": 4648,
        # "FR": 6551,
        # "PT": 1912,
        # "RU": 4385,
}

# cut_n_dict_3 = {
#         'cora':2485,
#         'cora_ml':2810,
#         'citeseer':2110,
#         'polblogs':1222,
#         'pubmed':19717,
#
#         'acm':3025,# 修改
#         'blogcatalog': 1000,
#         'uai':1800,# 修改
#         'flickr':500,
#
#         'cs':10050,
#         'physics':8000,
#         'dblp':8000,
#         'computers':5000,
#         'photo':4000,
#
#         'lastfmasia':7624,
#         'deezereurope':10000,
#         'facebookpagepage':10000,
#         'github':10000,
#
#         'usa':1190,
#         'brazil':131,
#         'europe':399,
#
# }
# compelte_n_dict={
#         'cora':2485,
#         'cora_ml':2810,
#         'citeseer':2110,
#         'polblogs':1222,
#         'pubmed':19717,
#
#         'acm':3025 ,
#         'blogcatalog': 3067,
#
#         'uai':5159,
#         'flickr':7575,
#
#         'cs':18333,
#         'physics':34493,
#         'dblp':17716,
#         'computers': 13752,
#         'photo':7650,
#
#         'lastfmasia': 7624,
#         'deezereurope': 28281,
#         'facebookpagepage': 22470,
#         'github': 37700,
#
#         'usa': 1190,
#         'brazil': 131,
#         'europe': 399,
#
#         'DE':9498,
#         "EN":7126,
#         "ES":4648,
#         "FR":6551,
#         "PT":1912,
#         "RU":4385,
# }
#
# dataset_n_class={
#         'cora':7,
#         'cora_ml':7,
#         'citeseer':6,
#         'polblogs':2,
#         'pubmed':3,
#
#         'acm':3,
#         'uai':19,
#         'blogcatalog': 6,
#         'flickr':9,
#
#         'cs':15,
#         'physics':5,
#         'dblp':4,
#         'computers':10,
#         'photo':8,
# }



