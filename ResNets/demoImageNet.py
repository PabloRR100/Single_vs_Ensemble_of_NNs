#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:31:56 2018
@author: pabloruizruiz
@title: Deep ResNet vs Shallow ResNets Ensemble on ImageNet
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
root = '/Users/pabloruizruiz/Harvard/Single_Ensembles/ResNets'
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

batch_size = 256
n_iters = 600000
n_epochs = int(n_iters / batch_size)

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

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

resnet18 = resnet18()
resnet34 = resnet34()
resnet50 = resnet50()
resnet101 = resnet101()
resnet152 = resnet152()


def parameters(model, typ=None):
    def compare_to_simplest(model, typ):
        simplest = count_parameters(resnet18)
        if typ is None: return count_parameters(model) / simplest
    return count_parameters(model)*1e-6, compare_to_simplest(model, typ)


table = BT()
table.append_row(['Model', 'M. Paramars', '% over ResNet20'])
table.append_row(['ResNset20', *parameters(resnet18)])
table.append_row(['ResNset32', *parameters(resnet34)])
table.append_row(['ResNset44', *parameters(resnet50)])
table.append_row(['ResNset56', *parameters(resnet101)])
table.append_row(['ResNset110', *parameters(resnet152)])
if comments: print(table)



# Apply constraint - Parameters constant

small = resnet18
big = resnet101 if ensemble_type == 'Big' else resnet152
ensemble_size = round(count_parameters(big) / count_parameters(small))


# Construct the ensemble

ensemble = []
for i in range(ensemble_size):
    model = resnet18()
    model.name = model.name + '_' + str(i+1)
    ensemble.append(model)


singleModel = resnet101 if ensemble_type == 'Big' else resnet152



# 3 - Train ResNet
# ----------------

from train import train
model = singleModel
criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      momentum=momentum, weight_decay=weight_decay)


# Big Single Model

singleModel.train()
single_history, single_time = train('ImageNet', singleModel, optimizer, criterion, train_loader,
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
        
        


