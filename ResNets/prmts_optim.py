#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:20:42 2019
@author: pabloruizruiz
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
sys.path.append('..')
sys.path.append('ResNets')
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


## DEVICES

cuda = torch.cuda.is_available()
n_workers = torch.multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = True if torch.cuda.device_count() > 1 else False
mem = False if device == 'cpu' else True


## PATHS

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


## MODELS

from resnets_Paper import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110

resnet20 = ResNet20()
resnet32 = ResNet32()
resnet44 = ResNet44()
resnet56 = ResNet56()
resnet110 = ResNet110()

## PICK MODEL
model = ResNet56()
model.to(device)
cudnn.benchmark = True
if gpus: model = nn.DataParallel(model)



def train(num_epoch, batch_size, learning_rate, momentum, weight_decay):    
    ''' Accept parameters and return the objective to minimize - loss '''
    
    ## DATA 
    
    # Dataloaders as a function of the batch size
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import transforms
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    train_dataset = CIFAR10(root = root, download = True, train = True, transform = train_transform)
    
    shuffle = True
    valid_size = 0.1
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(2019)
        np.random.shuffle(indices)
    
    train_idx = indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    ## QUESTION: Do we want to minimize training loss or validation accuracy?


    ## TRAINING STEP
    
    criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    
    for epoch in range(1, num_epoch+1):
        
        # Training
        for i, (images, labels) in enumerate(trainloader):
            
            images, labels = Variable(images), Variable(labels)            
            images, labels = images.to(device), labels.to(device)
            
            model.zero_grad()
            outputs = model(images)
            scores, predictions = torch.max(outputs.data, 1)
        
            loss = criterion(outputs, labels)            
            loss.backward()
            optimizer.step()

#            correct, total = 0, 0
#            total += outputs.size(0)
#            correct += int(sum(predictions == labels)) 
#            accuracy = correct / total
 
    # return -accuracy                 
    return loss.item()
    
    
def main(job_id, params):
    
    print(params)
    return train(params['num_epoch'],
                 params['batch_size'],
                 params['learning_rate'],
                 params['momemtum'],
                 params['weight_decay'])
    


