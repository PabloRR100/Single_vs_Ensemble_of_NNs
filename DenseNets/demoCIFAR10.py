#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:31:56 2018
@author: pabloruizruiz
@title: Deep DenseNet vs Shallow DenseNets Ensemble on CIFAR-10
"""

root = '/Users/pabloruizruiz/Harvard/Single_Ensembles/DenseNets'


import os
import multiprocessing
from beautifultable import BeautifulTable as BT

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

os.chdir(root)
path_to_logs = os.path.join(root, 'logs')
path_to_data = os.path.join(root, '../data')
path_to_output = os.path.join(root, 'outputs')
path_to_figures = os.path.join(path_to_output, 'figures')

import sys
sys.path.append('..')
from utils import load_dataset, count_parameters, figures



''' CONFIGURATION '''

draws = False               # Activate showing the figures
save_every = 1              # After how many epochs save stats
print_every = 2             # After how many epochs print stats
comments = True             # Activate printing comments
ensemble_type = 'Big'       # Single model big 
#ensemble_type = 'Huge'     # Single model huge

momentum = 0.9
weight_decay = 1e-4
learning_rate = 0.1

n_epochs = 40
batch_size = 64
n_iters = int(n_epochs * batch_size)

cuda = torch.cuda.is_available()
n_workers = multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# 1 - Import the Dataset
# ----------------------

path = os.path.join(root, 'data')
train_set, valid_set, test_set = load_dataset(path_to_data, 'CIFAR10', comments=comments)

train_loader = DataLoader(dataset = train_set.dataset, 
                               sampler=SubsetRandomSampler(train_set.indices),
                               batch_size = batch_size, num_workers=n_workers)

valid_loader = DataLoader(dataset = valid_set.dataset, 
                               sampler=SubsetRandomSampler(valid_set.indices),
                               batch_size = batch_size, num_workers=n_workers)

test_loader = DataLoader(dataset = test_set, batch_size = 1,
                               shuffle = False, num_workers=n_workers)




# 2 - Import the DenseNets
# ------------------------

from densenetsCIFAR10 import (denseNet_40_12, denseNet_100_12, denseNet_100_24, 
                              denseNetBC_100_12, denseNetBC_250_24, denseNetBC_190_40)

densenet_40_12 = denseNet_40_12()
densenet_100_12 = denseNet_100_12()
densenet_100_24 = denseNet_100_24()
densenetBC_100_12 = denseNetBC_100_12() 
densenetBC_250_24 = denseNetBC_250_24()
densenetBC_190_40 = denseNetBC_190_40()


def parameters(model, typ=None):
    def compare_to_simplest(model, typ):
        simplest1 = count_parameters(densenet_40_12)
        simplest2 = count_parameters(densenetBC_100_12)
        if typ is None: return count_parameters(model) / simplest1
        if typ == 'BC': return count_parameters(model) / simplest2
    return count_parameters(model)*1e-6, compare_to_simplest(model, typ)

table = BT()
table.append_row(['Model', 'k', 'L', 'M. of Params', '% Over simplest'])
table.append_row(['DenseNet', 12, 40, *parameters(densenet_40_12)])
table.append_row(['DenseNet', 12, 100, *parameters(densenet_100_12)])
table.append_row(['DenseNet', 24, 100, *parameters(densenet_100_24)])
table.append_row(['DenseNet-BC', 12, 100, *parameters(densenetBC_100_12, 'BC')])
table.append_row(['DenseNet-BC', 24, 250, *parameters(densenetBC_250_24, 'BC')])
table.append_row(['DenseNet-BC', 40, 190, *parameters(densenetBC_190_40, 'BC')])
if comments: print(table)


# Apply constraint - Parameters constant

small = count_parameters(denseNetBC_100_12())  # 19:1 vs 33:1
singleModel = denseNetBC_250_24() if ensemble_type == 'Big' else denseNetBC_190_40() 
ensemble_size = round(count_parameters(singleModel) / small)


# Construct the ensemble

ensemble = []

for i in range(ensemble_size):
    model = denseNetBC_100_12()
    model.name = model.name + '_' + str(i+1)
    ensemble.append(model)



# 3 - Train DenseNet
# ------------------

from train import train
criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      momentum=momentum, weight_decay=weight_decay)


# Big Single Model

singleModel.train()
single_history, single_time = train('CIFAR10', singleModel, optimizer, criterion, train_loader,
                                    n_epochs, n_iters, createlog=False, logpath=None, 
                                    print_every=print_every, save_frequency=save_every)

figures(single_history, singleModel.name, 'CIFAR10', path_to_figures, draws)
single_history.to_csv(os.path.join(path_to_output, singleModel.name + '.csv'))


# Ensemble individuals

ensemble_history = []
for model in ensemble:
    model.train()
    model_history, model_time = train('CIFAR10', model, optimizer, criterion, train_loader, 
                                      n_epochs, n_iters, createlog=False, logpath=None, 
                                      print_every=print_every, save_frequency=save_every)
    ensemble_history.append((model_history, model_time))
    
for i, model in enumerate(ensemble):
    model_history, model_time = ensemble_history[i]
    figures(model_history, model.name, 'CIFAR10', path_to_figures, draws)
    model_history.to_csv(os.path.join(path_to_output, model.name + '.csv'))




# 4 - Evaluate Models
# -------------------

# Big Single Model
    
singleModel.eval()
total, correct = 0,0
with torch.no_grad():
    
    for i, (images, labels) in enumerate(test_loader):
        
        images = Variable(images)
        labels = Variable(labels)
        
        outputs = singleModel(images)
        
        _, preds = outputs.max(1)
        total += outputs.size(0)
        correct += int(sum(preds == labels))
        
        if (i % 50 == 0 and (i < 100 or (i > 1000 and i < 1050))) or i % 1000 == 0:
            print('Image [{}/{}]. Total correct {}'.format(i,len(test_set),correct))                
    
    print('Accuracy of the network on the {} test images: {}%'.format(len(test_set), (100 * correct / total)))
     
        
# Ensemble Model
    
total, correct = 0,0
with torch.no_grad():
    
    for i, (images, labels) in enumerate(test_loader):
        
        #images, labels = test_set[0]
        images = Variable(images)
        labels = Variable(torch.tensor(labels))        
        
        outputs = []
        for model in ensemble:
            
            model.eval()
            output = model(images)
            outputs.append(output)
            
        outputs = torch.mean(torch.stack(outputs), dim=0)
            
        _, preds = outputs.max(1)
        total += outputs.size(0)
        correct += int(sum(preds == labels))
        
        if (i % 50 == 0 and (i < 100 or (i > 1000 and i < 1050))) or i % 1000 == 0:
            print('Image [{}/{}]. Total correct {}'.format(i,len(test_set),correct))    
            
    print('Accuracy of the network on the {} test images: {}%'.format(len(test_set), (100 * correct / total)))
        
        


