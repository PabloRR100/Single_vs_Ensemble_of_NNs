 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:31:56 2018
@author: pabloruizruiz
@title: Deep ResNets vs Shallow ResNets Ensemble on CIFAR-10
"""

import os
import glob
import pickle
from beautifultable import BeautifulTable as BT

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import sys
sys.path.append('..')
sys.path.append('ResNets')
from utils import load_dataset, count_parameters


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'ImportWarning')
warnings.filterwarnings('ignore', 'DeprecationWarning')


''' 
CONFIGURATION 
-------------
'''
print('\n\nCONFIGURATION')
print('-------------')


dataset = 'CIFAR10'
comments = True             # Activate printing comments
ensemble_type = 'Big'       # Single model big 

n_epochs = 200
batch_size = 128

momentum = 0.9
learning_rate = 0.1
weight_decay = 1e-4

load_trained_models = False



cuda = torch.cuda.is_available()
n_workers = torch.multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = True if torch.cuda.device_count() > 1 else False
mem = False if device == 'cpu' else True


table = BT()
table.append_row(['Python Version', sys.version[:5]])
table.append_row(['PyTorch Version', torch.__version__])
table.append_row(['Cuda', str(cuda)])
table.append_row(['Device', str(device)])
table.append_row(['Cores', str(n_workers)])
table.append_row(['GPUs', str(torch.cuda.device_count())])
table.append_row(['CUDNN Enabled', str(torch.backends.cudnn.enabled)])
print('\n\nCOMPUTING CONFIG')
print('----------------')
print(table)




'''
DEFININTION OF PATHS 
--------------------
Define all the paths to load / save files
Ensure all those paths are correctly defined before moving on
'''

print('DEFINITION OF PATHS')
print('-------------------')
scripts = os.getcwd()
root = os.path.abspath(os.path.join(scripts, '../'))
results = os.path.abspath(os.path.join(root, 'results'))
data_path = os.path.abspath(os.path.join(root, '../datasets'))

path_to_logs = os.path.join(results, 'logs', 'resnets')
path_to_models = os.path.join(results, 'models', 'resnets')
path_to_figures = os.path.join(results, 'figures', 'resnets')
path_to_definitives = os.path.join(path_to_models, 'definitives')

train_log = os.path.join(path_to_logs, 'train')
test_log = os.path.join(path_to_logs, 'test')

assert os.path.exists(root), 'Root folder not found'
assert os.path.exists(scripts), 'Scripts folder not found'
assert os.path.exists(results), 'Results folder not found'
assert os.path.exists(data_path), 'Data folder not found'
assert os.path.exists(path_to_logs), 'Logs folder not found'
assert os.path.exists(path_to_models), 'Models folder not found'
assert os.path.exists(path_to_figures), 'Figure folder not found'
assert os.path.exists(path_to_definitives), 'Def. models folder not found'

print('Paths Validated')
print('---------------')
print('Root path: ', root)
print('Script path: ', scripts)
print('Results path: ', results)
print('DataFolder path: ', data_path)
print('Models to save path: ', path_to_models)
print('Models to load path: ', path_to_definitives)

paths = {
    'root': root, 
    'script': scripts,
    'data': data_path,
    'resulsts': results,
    'logs': {'train': train_log, 'test': test_log}, 
    'models': path_to_models,
    'definitives': path_to_definitives,
    'figures': path_to_figures
}



# 1 - Import the Dataset
# ----------------------

print('IMPORTING DATA')
print('--------------')

from data import create_data_loaders
train_loader, valid_loader = create_data_loaders(batch_size, n_workers)


# 2 - Import the ResNet
# ---------------------

print('\n\nLOADING MODELS')
print('----------------')

from resnets import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110

resnet20 = ResNet20()
resnet32 = ResNet32()
resnet44 = ResNet44()
resnet56 = ResNet56()
resnet110 = ResNet110()

def parameters(model, typ=None):
    def compare_to_simplest(model, typ):
        simplest = count_parameters(resnet20)
        if typ is None: return count_parameters(model) / simplest
    return count_parameters(model)*1e-6, compare_to_simplest(model, typ)


table = BT()
table.append_row(['Model', 'M. Paramars', '% over ResNet20'])
table.append_row(['ResNet 20', *parameters(resnet20)])
table.append_row(['ResNet 32', *parameters(resnet32)])
table.append_row(['ResNet 44', *parameters(resnet44)])
table.append_row(['ResNet 56', *parameters(resnet56)])
table.append_row(['ResNet 110', *parameters(resnet110)])
if comments: print(table)


# Apply constraint - Parameters constant

small = count_parameters(ResNet20())  # 3:1 vs 6:1
singleModel = ResNet56() if ensemble_type == 'Big' else ResNet110() 
ensemble_size = round(count_parameters(singleModel) / small)


# Construct the single model

singleModel = ResNet56() if ensemble_type == 'Big' else ResNet110() # 3:1 vs 6:1
title = singleModel.name

name = singleModel.name
optimizer = optim.SGD(singleModel.parameters(), learning_rate, momentum, weight_decay)


# Construct the ensemble

names = []
ensemble = []
optimizers = []
for i in range(ensemble_size):
    
    model = ResNet20()
    names.append(model.name + '_' + str(i+1))
    params = optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    optimizers.append(params)
    
    model.to(device)
    if gpus: model = nn.DataParallel(model)
    ensemble.append(model)



# 3 - Train ResNet
# ----------------

print('\n\nTRAINING')
print('--------')

criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()


# Big Single Model

from train import train
print('Starting Single Model Training...' )

n_iters = n_epochs * len(train_loader)
params = [dataset, name, singleModel, optimizer, criterion, device, train_loader, 
          valid_loader, n_epochs, n_iters, paths]

results = train(*params)
with open('Results_Single_Models.pkl', 'wb') as object_result:
    pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL)

results.show()


# Ensemble Model

from train_ensemble import train as train_ensemble
print('Starting Ensemble Training...')

params = [dataset, names, ensemble, optimizers, criterion, device, train_loader,
          valid_loader, n_epochs, n_iters, paths]
    
ens_results = train_ensemble(*params)
with open('Results_Ensemble_Models.pkl', 'wb') as object_result:
    pickle.dump(ens_results, object_result, pickle.HIGHEST_PROTOCOL)

ens_results.show()



# 4 - Evaluate Models
# -------------------
    
print('\n\nTESTING')
print('-------')

from test import test

testresults = test('CIFAR10', name, singleModel, ensemble, device, valid_loader, paths)
with open('Results_Testing.pkl', 'wb') as object_result:
    pickle.dump(testresults, object_result, pickle.HIGHEST_PROTOCOL)



exit()




#if not True: ## Just to avoid running this from Azure - it breaks
#    
#    # Training figures
#    with open('Results_Single_Models.pkl', 'rb') as input:
#        res = pickle.load(input)
#    
#    with open('Results_Ensemble_Models.pkl', 'rb') as input:
#        eres = pickle.load(input)
#    
#    
#    data1 = {'single':res.iter_train_accy, 
#            'ensemble': eres.iter_train_accy['ensemble']}
#    
#    data2 = {'single':res.iter_train_loss, 
#            'ensemble': eres.iter_train_loss['ensemble']}
#    
#    data3 = {'single':res.train_accy, 
#            'ensemble': eres.train_accy['ensemble']}
#    
#    data4 = {'single':res.train_loss, 
#            'ensemble': eres.train_loss['ensemble']}
#    
#    data5 = {'single':res.valid_accy, 
#            'ensemble': eres.valid_accy['ensemble']}
#    
#    data6 = {'single':res.valid_loss, 
#            'ensemble': eres.valid_loss['ensemble']}
#     
#    import pandas as pd
#    import seaborn as sns
#    sns.lineplot(data=pd.DataFrame.from_dict(eres.valid_loss))
#    
#    sns.lineplot(data=pd.DataFrame.from_dict(eres.valid_loss))
#    
#    savefig(data1, path_to_figures, title + '_train_accuracy_per_iter.png')
#    savefig(data2, path_to_figures, title + '_train_loss_per_iter.png')
#    savefig(data3, path_to_figures, title + '_train_accuracy_per_epoch.png')
#    savefig(data4, path_to_figures, title + '_train_loss_per_iter.png')
#    savefig(data5, path_to_figures, title + '_valid_accuracy.png')
#    savefig(data6, path_to_figures, title + '_valid_loss.png')


