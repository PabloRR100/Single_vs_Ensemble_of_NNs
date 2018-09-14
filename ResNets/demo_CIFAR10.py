#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:31:56 2018
@author: pabloruizruiz
@title: Deep ResNets vs Shallow ResNets Ensemble on CIFAR-10
"""

import os
import pickle
import multiprocessing
from beautifultable import BeautifulTable as BT


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


import sys
sys.path.append('..')
sys.path.append('ResNets')
from utils import def_training, load_dataset, count_parameters, figures


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'ImportWarning')
warnings.filterwarnings('ignore', 'DeprecationWarning')


''' 
CONFIGURATION 
-------------

Catch from the parser all the parameters to define the training
'''
print('\n\nCONFIGURATION')
print('-------------')

#from parser import args
#
#save = args.save
#name = args.name
#draws = args.draws
#dataset = args.dataset
#testing = args.testing
#comments = args.comments
#
#ensemble_type = args.ensembleSize
#
#n_epochs = args.epochs
#n_iters = args.iterations
#batch_size = args.batch_size
#learning_rate = args.learning_rate
#save_frequency = args.save_frequency
#
#if args.name is None: args.name = 'ResNet'
## Sanity check for epochs - batch size - iterations
#n_iters, n_epochs, batch_size = def_training(n_iters, n_epochs, batch_size)
#
## Display config to run file
#table = BT()
#table.append_row(['Save', str(args.save)])
#table.append_row(['Name', str(args.name)])
#table.append_row(['Draws', str(args.draws)])
#table.append_row(['Testing', str(args.testing)])
#table.append_row(['Comments', str(args.comments)])
#table.append_row(['Ensemble size', str(args.ensembleSize)])
#table.append_row(['-------------', '-------------'])
#table.append_row(['Epochs', n_epochs])
#table.append_row(['Iterations', n_iters])
#table.append_row(['Batch Size', batch_size])
#table.append_row(['Learning Rate', str(args.learning_rate)])
#print(table)
#

#######################################################
# Backup code to debug from python shell - no parser
save = False                # Activate results saving 
draws = False               # Activate showing the figures
dataset = 'CIFAR10'
testing = True             # Activate test to run few iterations per epoch       
comments = True             # Activate printing comments
createlog = False           # Activate option to save the logs in .txt
save_frequency = 1          # After how many epochs save stats
ensemble_type = 'Big'       # Single model big 
#ensemble_type = 'Huge'     # Single model huge
learning_rate = 0.1
batch_size = 128
n_iters = 64000
#######################################################


momentum = 0.9
weight_decay = 1e-4

n_epochs = int(n_iters / batch_size)

# GPU if CUDA is available
cuda = torch.cuda.is_available()
n_workers = multiprocessing.cpu_count()
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
path_to_dataframes = os.path.join(results, 'dataframes', 'resnets')

train_log = os.path.join(path_to_logs, 'train')
test_log = os.path.join(path_to_logs, 'test')

assert os.path.exists(root), 'Root folder not found'
assert os.path.exists(scripts), 'Scripts folder not found'
assert os.path.exists(results), 'Results folder not found'
assert os.path.exists(data_path), 'Data folder not found'
assert os.path.exists(path_to_logs), 'Logs folder not found'
assert os.path.exists(path_to_models), 'Models folder not found'
assert os.path.exists(path_to_figures), 'Figure folder not found'
assert os.path.exists(path_to_dataframes), 'Dataframes folder not found'

print('Paths Validated')
print('---------------')
print('Root path: ', root)
print('Script path: ', scripts)
print('Result path: ', results)
print('DataFolder path: ', data_path)

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



# 1 - Import the Dataset
# ----------------------

print('IMPORTING DATA')
print('--------------')

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



# 2 - Import the ResNet
# ---------------------

print('\n\nIMPORTING MODELS')
print('----------------')

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
if comments: print(table)


# Apply constraint - Parameters constant

small = count_parameters(ResNet20())  # 3:1 vs 6:1
singleModel = ResNet56() if ensemble_type == 'Big' else ResNet110() 
ensemble_size = round(count_parameters(singleModel) / small)


# Construct the single model
singleModel = ResNet56() if ensemble_type == 'Big' else ResNet110() # 3:1 vs 6:1

name = singleModel.name
singleModel.to(device)
if gpus: singleModel = nn.DataParallel(singleModel)


# Construct the ensemble

names = []
ensemble = []
optimizers = []
for i in range(ensemble_size):
    
    model = ResNet20()
    names.append(model.name + '_' + str(i+1))
    params = [optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)]
    optimizers.append(*params)
    
    model.to(device)
    if gpus: model = nn.DataParallel(model)
    ensemble.append(model)




# 3 - Train ResNet
# ----------------

print('\n\nTRAINING')
print('--------')

from train import train
from train_ensemble import train as train_ensemble
criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()

# Big Single Model
#
#optimizer = optim.SGD(singleModel.parameters(), learning_rate, momentum, weight_decay)
#
#print('Starting Single Model Training...' )
#params = [dataset, name, singleModel, optimizer, criterion, device, train_loader,
#          valid_loader, n_epochs, n_iters, save, paths, save_frequency, testing]
#
#results, timer = train(*params)
#
#results = train_ensemble(*params)
#with open('Results_Single_Model.pkl', 'wb') as result:
#    pickle.dump(results, result, pickle.HIGHEST_PROTOCOL)


#figures(train_history, 'train_' + name, dataset, paths['figures'], draws, save)
#figures(valid_history, 'valid_' + name, dataset, paths['figures'], draws, save)
#if save: train_history.to_csv(os.path.join(paths['dataframes'], 'train_' + name + '.csv'))
#if save: valid_history.to_csv(os.path.join(paths['dataframes'], 'valid_' + name + '.csv'))


# Ensemble Model

print('Starting Ensemble Training...')

params = [dataset, names, ensemble, optimizers, criterion, device, train_loader,
          valid_loader, n_epochs, n_iters, save, paths, save_frequency, testing]
    
results, timer = train_ensemble(*params)
with open('Results_Ensemble_Models.pkl', 'wb') as result:
    pickle.dump(results, result, pickle.HIGHEST_PROTOCOL)

results.show()

## Training figures
#with open('Results_Ensemble_Models.pkl', 'rb') as input:
#    results = pickle.load(input)


#
## 4 - Evaluate Models
## -------------------
#    
#print('\n\nTESTING')
#print('-------'); bl
#
#from test import test
#test('CIFAR10', name, singleModel, ensemble, device, test_loader, paths, save)

results.show()

exit()


