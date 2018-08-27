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


import sys
sys.path.append('..')


''' DEFININTION OF PATHS '''
scripts = os.getcwd()
root = os.path.abspath(os.path.join(scripts, '../'))
results = os.path.abspath(os.path.join(root, 'results'))
data_path = os.path.abspath(os.path.join(root, '../datasets'))

path_to_logs = os.path.join(results, 'logs')
path_to_models = os.path.join(results, 'models')
path_to_figures = os.path.join(results, 'figures')
path_to_dataframes = os.path.join(results, 'dataframes')

assert os.path.exists(root), 'Root folder not found'
assert os.path.exists(scripts), 'Scripts folder not found'
assert os.path.exists(results), 'Results folder not found'
assert os.path.exists(data_path), 'Data folder not found'
assert os.path.exists(path_to_logs), 'Logs folder not found'
assert os.path.exists(path_to_models), 'Models folder not found'
assert os.path.exists(path_to_figures), 'Figure folder not found'
assert os.path.exists(path_to_dataframes), 'Dataframes folder not found'

import warnings
warnings.filterwarnings("ignore")
from utils import load_dataset, count_parameters, figures



''' CONFIGURATION '''
#from parser import args
#
#save = args.save
#name = args.name
#test = args.test
#draws = args.draws
#dataset = 'CIFAR10'
#comments = args.comments
#
#ensemble_type = args.ensembleSize
#
#n_iters = args.iterations
#batch_size = args.batch_size
#learning_rate = args.learning_rate

save = False                # Activate results saving 
test = True                 # Activate test to run few iterations per epoch       
draws = False               # Activate showing the figures
print_every = 2             # After how many epochs print stats
comments = True             # Activate printing comments
createlog = False           # Activate option to save the logs in .txt
save_frequency = 1          # After how many epochs save stats
ensemble_type = 'Big'       # Single model big 
#ensemble_type = 'Huge'     # Single model huge
learning_rate = 0.1
batch_size = 128
n_iters = 64000

momentum = 0.9
weight_decay = 1e-4

n_epochs = int(n_iters / batch_size)

cuda = torch.cuda.is_available()
n_workers = multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# 1 - Import the Dataset
# ----------------------

train_set, valid_set, test_set = load_dataset(data_path, 'CIFAR10', comments=comments)

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


# Construct the ensemble

ensemble = []

for i in range(ensemble_size):
    model = ResNet20()
    model.name = model.name + '_' + str(i+1)
    ensemble.append(model)

    
singleModel = ResNet56() if ensemble_type == 'Big' else ResNet110() # 3:1 vs 6:1




# 3 - Train ResNet
# ----------------

from train import train
train_log = os.path.join(path_to_logs, 'train')
criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      momentum=momentum, weight_decay=weight_decay)


# Big Single Model

singleModel.train()
single_history, single_time = train('CIFAR10', singleModel, optimizer, criterion, train_loader,
                                    n_epochs, n_iters, save, createlog, None, print_every, save_frequency)

figures(single_history, singleModel.name, 'CIFAR10', path_to_figures, draws, save)
if save: single_history.to_csv(os.path.join(path_to_dataframes, singleModel.name + '.csv'))


# Ensemble individuals

ensemble_history = []
for model in ensemble:
    model.train()
    model_history, model_time = train('CIFAR10', model, optimizer, criterion, train_loader, 
                                      n_epochs, n_iters, save, createlog, None, print_every, save_frequency)
    ensemble_history.append((model_history, model_time))

for i, model in enumerate(ensemble):  
    model_history, model_time = ensemble_history[i]
    figures(model_history, model.name, 'CIFAR10', path_to_figures, draws, save)
    if save: model_history.to_csv(os.path.join(path_to_dataframes, model.name + '.csv'))




# 4 - Evaluate Models
# -------------------

from test import test
test_log = os.path.join(path_to_logs, 'test')    

test('CIFAR10', singleModel, ensemble, test_loader, test_log)


exit()



#singleModel.eval()
#total, correct = 0,0
#with torch.no_grad():
#    
#    for i, (images, labels) in enumerate(test_loader):
#        
#        images = Variable(images)
#        labels = Variable(labels)
#        
#        outputs = singleModel(images)
#        
#        _, preds = outputs.max(1)
#        total += outputs.size(0)
#        correct += int(sum(preds == labels))
#        
#        if (i % 50 == 0 and (i < 100 or (i > 1000 and i < 1050))) or i % 1000 == 0:
#            print('Image [{}/{}]. Total correct {}'.format(i,len(test_set),correct))                
#    
#    print('Accuracy of the network on the {} test images: {}%'.format(len(test_set), (100 * correct / total)))
#     
#        
## Ensemble Model
#    
#total, correct = 0,0
#with torch.no_grad():
#    
#    for i, (images, labels) in enumerate(test_loader):
#        
#        #images, labels = test_set[0]
#        images = Variable(images)
#        labels = Variable(torch.tensor(labels))        
#        
#        outputs = []
#        for model in ensemble:
#            
#            model.eval()
#            output = model(images)
#            outputs.append(output)
#            
#        outputs = torch.mean(torch.stack(outputs), dim=0)
#            
#        _, preds = outputs.max(1)
#        total += outputs.size(0)
#        correct += int(sum(preds == labels))
#        
#        if (i % 50 == 0 and (i < 100 or (i > 1000 and i < 1050))) or i % 1000 == 0:
#            print('Image [{}/{}]. Total correct {}'.format(i,len(test_set),correct))                
#            
#    print('Accuracy of the network on the {} test images: {}%'.format(len(test_set), (100 * correct / total)))


# BACKUP CODE

#save = False                # Activate results saving 
#test = True                 # Activate test to run few iterations per epoch       
#draws = False               # Activate showing the figures
#print_every = 2             # After how many epochs print stats
#comments = True             # Activate printing comments
#createlog = False           # Activate option to save the logs in .txt
#save_frequency = 1          # After how many epochs save stats
#ensemble_type = 'Big'       # Single model big 
##ensemble_type = 'Huge'     # Single model huge
#
#momentum = 0.9
#weight_decay = 1e-4
#learning_rate = 0.1
#
#batch_size = 128
#n_iters = 64000
#n_epochs = int(n_iters / batch_size)
