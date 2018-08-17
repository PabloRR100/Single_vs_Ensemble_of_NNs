#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:31:56 2018
@author: pabloruizruiz
@title: Deep DenseNet vs Shallow DenseNets Ensemble on ImageNet
"""

import os
import multiprocessing
from beautifultable import BeautifulTable as BT


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


import sys
sys.path.append('..')
data = '/Users/pabloruizruiz/Harvard/Single_Ensembles/data'
root = '/Users/pabloruizruiz/Harvard/Single_Ensembles/DenseNets'
results = '/Users/pabloruizruiz/Harvard/Single_Ensembles/results'

os.chdir(root)
path_to_data = data
path_to_logs = os.path.join(results, 'logs', 'resnets')
path_to_figures = os.path.join(results, 'figures', 'resnets')
path_to_outputs = os.path.join(results, 'dataframes', 'resnets')


import warnings
warnings.filterwarnings("ignore")
from utils import load_dataset, count_parameters, figures



''' CONFIGURATION '''

test = True                 # Activate test to run few iterations per epoch       
draws = False               # Activate showing the figures
save_every = 1              # After how many epochs save stats
print_every = 2             # After how many epochs print stats
comments = True             # Activate printing comments
ensemble_type = 'Big'       # Single model big 
#ensemble_type = 'Huge'     # Single model huge

momentum = 0.9
weight_decay = 1e-4
learning_rate = 0.1

n_epochs = 90
batch_size = 256
n_iters = int(n_epochs * batch_size)

cuda = torch.cuda.is_available()
n_workers = multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# 1 - Import the Dataset
# ----------------------

path = os.path.join(root, 'data')
train_set, valid_set, test_set = load_dataset(path_to_data, 'ImageNet', comments=comments)

train_loader = DataLoader(dataset = train_set.dataset, 
                               sampler=SubsetRandomSampler(train_set.indices),
                               batch_size = batch_size, num_workers=n_workers)

valid_loader = DataLoader(dataset = valid_set.dataset, 
                               sampler=SubsetRandomSampler(valid_set.indices),
                               batch_size = batch_size, num_workers=n_workers)

test_loader = DataLoader(dataset = test_set, batch_size = 1,
                               shuffle = False, num_workers=n_workers)




# 2 - Import the ResNet
# ---------------------

from torchvision.models import densenet121, densenet169, densenet201

densenet121 = densenet121()
densenet169 = densenet169()
densenet201 = densenet201()

table = BT()
table.append_row(['Model', 'M. Paramars', '% over ResNet20'])
table.append_row(['DenseNet121', count_parameters(densenet121)/1e6, 1])
table.append_row(['DenseNet169', count_parameters(densenet169)/1e6, 
                  count_parameters(densenet169)/count_parameters(densenet121)])
table.append_row(['DenseNet201', count_parameters(densenet201)/1e6, 
                  count_parameters(densenet201)/count_parameters(densenet121)])
if comments: print(table)


# Apply constraint - Parameters constant

small = densenet121
big = densenet169
sup = densenet201
ensemble_size = round(count_parameters(big) / count_parameters(small))
ensemble_size_sup = round(count_parameters(sup) / count_parameters(small))


# Construct the ensemble

ensemble = []
superens = []
for i in range(ensemble_size):
    model = densenet121
    model.name = model.name + '_' + str(i+1)
    ensemble.append(model)

for i in range(ensemble_size_sup):
    model = densenet121()
    model.name = model.name + '_' + str(i+1)
    superens.append(model)
    
singleModel = densenet169
superModel = densenet201



# 3 - Train ResNet
# ----------------

from train import train
model = singleModel
criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      momentum=momentum, weight_decay=weight_decay)


# Big Single Model

singleModel.train()
single_history, single_time = train(singleModel, optimizer, criterion, train_loader,
                                    n_epochs, n_iters, createlog=False, logpath=None, 
                                    print_every=print_every, save_frequency=save_every)

figures(single_history, singleModel.name, 'ImageNet', path_to_figures, draws)
single_history.to_csv(os.path.join(path_to_outputs, singleModel.name + '.csv'))



# Ensemble individuals

ensemble_history = []
for model in ensemble:
    model.train()
    model_history, model_time = train('ImageNet', model, optimizer, criterion, train_loader, 
                                      n_epochs, n_iters, createlog=False, logpath=None, 
                                      print_every=print_every, save_frequency=save_every)
    ensemble_history.append((model_history, model_time))
    
    figures(model_history, model.name, 'ImageNet', path_to_figures, draws)
    model_history.to_csv(os.path.join(path_to_outputs, model.name + '.csv'))




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
        
        if i % 50 == 0:
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
        
        if i % 50 == 0:
            print('Image [{}/{}]. Total correct {}'.format(i,len(test_set),correct))    
            
    print('Accuracy of the network on the {} test images: {}%'.format(len(test_set), (100 * correct / total)))
        
        


