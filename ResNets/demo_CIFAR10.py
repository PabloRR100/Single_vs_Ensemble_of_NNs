#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:31:56 2018
@author: pabloruizruiz
@title: Deep ResNets vs Shallow ResNets Ensemble on CIFAR-10
"""

import os
import multiprocessing
from beautifultable import BeautifulTable as BT


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


bl = print('\n') 
import warnings
warnings.filterwarnings("ignore")


import sys
sys.path.append('..')
from utils import load_dataset, count_parameters, figures



''' 
CONFIGURATION 
-------------

Catch from the parser all the parameters to define the training
'''
print('CONFIGURATION')
print('-------------'); bl

from parser import args

save = args.save
name = args.name
draws = args.draws
dataset = args.dataset
testing = args.testing
comments = args.comments

ensemble_type = args.ensembleSize

n_iters = args.iterations
batch_size = args.batch_size
learning_rate = args.learning_rate
save_frequency = args.save_frequency

## Backup code to debug from python shell - no parser
#save = False                # Activate results saving 
#draws = False               # Activate showing the figures
#testing = True             # Activate test to run few iterations per epoch       
#comments = True             # Activate printing comments
#createlog = False           # Activate option to save the logs in .txt
#save_frequency = 1          # After how many epochs save stats
#ensemble_type = 'Big'       # Single model big 
##ensemble_type = 'Huge'     # Single model huge
#learning_rate = 0.1
#batch_size = 128
#n_iters = 64000

momentum = 0.9
weight_decay = 1e-4

n_epochs = int(n_iters / batch_size)

# GPU if CUDA is available
cuda = torch.cuda.is_available()
n_workers = multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = True if torch.cuda.device_count() > 1 else False
mem = False if device == 'cpu' else True


for arg in vars(args):
    print(arg, getattr(args, arg), type(arg))

bl
bl
sys
table = BT()
table.append_row(['Python Interpreter', os.system("which python")])
table.append_row(['Python Version', os.system('python -V')])
table.append_row(['Memory check (MBs)', os.system('free -m -h')])
table.append_row(['Cuda', str(cuda)])
table.append_row(['Device', str(device)])
table.append_row(['Cores', str(n_workers)])
table.append_row(['GPUs', str(torch.cuda.device_count())])
table.append_row(['CUDNN Enabled', str(torch.backends.cudnn.enabled)])
print(table)

bl
bl

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
path_to_dataframes = os.path.join(results, 'dataframes', 'resnets')

train_log = os.path.join(path_to_logs, 'train')
test_log = os.path.join(path_to_logs, 'test')

print('Root path: ', root)
print('Script path: ', scripts)
print('Result path: ', results)
print('DataFolder path: ', data_path)

assert os.path.exists(root), 'Root folder not found'
assert os.path.exists(scripts), 'Scripts folder not found'
assert os.path.exists(results), 'Results folder not found'
assert os.path.exists(data_path), 'Data folder not found'
assert os.path.exists(path_to_logs), 'Logs folder not found'
assert os.path.exists(path_to_models), 'Models folder not found'
assert os.path.exists(path_to_figures), 'Figure folder not found'
assert os.path.exists(path_to_dataframes), 'Dataframes folder not found'

paths = {
        'root': root, 
        'script': scripts,
        'data': data_path,
        'resulsts': results,
        'logs': {'train': train_log, 'test': test_log}, 
        'models': path_to_models,
        'figures': path_to_figures,
        'dataframes': path_to_dataframes
        }

bl
bl



# 1 - Import the Dataset
# ----------------------

print('IMPORTING DATA')
print('--------------'); bl

train_set, valid_set, test_set = load_dataset(data_path, dataset, comments=comments)

train_loader = DataLoader(dataset = train_set.dataset, 
                          sampler=SubsetRandomSampler(train_set.indices),
                          batch_size = batch_size, num_workers=n_workers,
                          pin_memory = mem)

valid_loader = DataLoader(dataset = valid_set.dataset, 
                          sampler=SubsetRandomSampler(valid_set.indices),
                          batch_size = batch_size, num_workers=n_workers,
                          pin_memory = mem)

test_loader = DataLoader(dataset = test_set, batch_size = 1,
                         shuffle = False, num_workers=n_workers, pin_memory = mem)


bl
bl


# 2 - Import the ResNet
# ---------------------

print('IMPORTING MODELS')
print('----------------'); bl

from resnets_CIFAR10 import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110

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
table.append_row(['ResNset20', *parameters(resnet20)])
table.append_row(['ResNset32', *parameters(resnet32)])
table.append_row(['ResNset44', *parameters(resnet44)])
table.append_row(['ResNset56', *parameters(resnet56)])
table.append_row(['ResNset110', *parameters(resnet110)])
if comments: bl; print(table)


# Apply constraint - Parameters constant

small = count_parameters(ResNet20())  # 3:1 vs 6:1
singleModel = ResNet56() if ensemble_type == 'Big' else ResNet110() 
ensemble_size = round(count_parameters(singleModel) / small)


# Construct the ensemble

ensemble = []

for i in range(ensemble_size):
    model = ResNet20()
    model.name = model.name + '_' + str(i+1)
    ensemble.append(model)

    
singleModel = ResNet56() if ensemble_type == 'Big' else ResNet110() # 3:1 vs 6:1


bl
bl


# 3 - Train ResNet
# ----------------

print('TRAINING')
print('--------'); bl

from train import train

criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      momentum=momentum, weight_decay=weight_decay)


# Big Single Model

name = singleModel.name
singleModel.train()
if gpus: 
    singleModel = nn.DataParallel(singleModel)
singleModel.to(device)

single_history, single_time = train('CIFAR10', singleModel, optimizer, criterion, device, train_loader,
                                    n_epochs, n_iters, save, paths, save_frequency, testing)

figures(single_history, name, 'CIFAR10', paths['figures'], draws, save)
if save: single_history.to_csv(os.path.join(paths['dataframes'], singleModel.name + '.csv'))


# Ensemble individuals

names = []
ensemble_history = []
for model in ensemble:
    names.append(model.name)
    model.train()
    if gpus: 
        model = nn.DataParallel(model)
    model.to(device)
    model_history, model_time = train('CIFAR10', model, optimizer, criterion, device, train_loader, 
                                      n_epochs, n_iters, save, paths, save_frequency, testing)
    ensemble_history.append((model_history, model_time))

for i, model in enumerate(ensemble):  
    model_history, model_time = ensemble_history[i]
    figures(model_history, names[i], 'CIFAR10', paths['figures'], draws, save)
    if save: model_history.to_csv(os.path.join(paths['dataframes'], model.name + '.csv'))


bl
bl


# 4 - Evaluate Models
# -------------------
    
print('TESTING')
print('-------'); bl

from test import test
test('CIFAR10', singleModel, ensemble, device, test_loader, paths, save)


exit()
