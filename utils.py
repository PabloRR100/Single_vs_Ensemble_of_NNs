#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:49:17 2018
@author: pabloruizruiz
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data.dataset import random_split


# DATASET 
# -------

# Define images preprocessing steps
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4), 
        transforms.ToTensor(), 
        normalize])


# Load Dataset
def load_dataset(data_path, dataset: str, comments: bool = True):
    
    assert os.path.exists(data_path), 'Datafolder not found'
    def dataset_info(train_dataset, valid_dataset, test_dataset, name):
        
        from beautifultable import BeautifulTable as BT
        if hasattr(test_dataset, 'classes'): classes = len(test_dataset.classes)
        elif hasattr(test_dataset, 'labels'): classes = len(np.unique(test_dataset.labels))
        elif hasattr(test_dataset, 'test_labels'): classes = len(np.unique(test_dataset.test_labels))
        else: print('Classes not detected in the dataset')
        
        table = BT()
        table.append_row(['Train Images', len(train_dataset.indices)])
        table.append_row(['Valid Images', len(valid_dataset.indices)])
        table.append_row(['Test Images', len(test_dataset)])
        table.append_row(['Classes', classes])
        print(table)
        
    if dataset == 'CIFAR10':
        transform = transforms
        root = os.path.join(data_path, 'CIFAR10')
        
        train_dataset = CIFAR10(root = root, download = True, train = True, transform = transform)
        test_dataset  = CIFAR10(root = root, download = False, train = False, transform = transform)
    
    
    len_ = len(train_dataset)
    train_dataset, valid_dataset = random_split(train_dataset, [round(len_*0.9), round(len_*0.1)])
    if comments: dataset_info(train_dataset, valid_dataset, test_dataset, name=dataset)
    return train_dataset, valid_dataset, test_dataset



# Plot and save output figures
def figures(data, name, dataset_name, path, draws):
        
    # Loss evolution
    image_file = 'baseline_' + name + '_' + dataset_name + '_training_loss.png'
    image_file = os.path.join(path, image_file)
    plt.figure()
    plt.title('Loss Evolution')
    plt.plot(data['Loss'], 'r-', label='Training')
    plt.legend()
    plt.savefig(image_file)
    if draws: plt.show()
    
    # Accuracy evolution
    plt.figure()
    image_file = 'baseline_' + name + '_' + dataset_name + '_training_accuracy.png'
    image_file = os.path.join(path, image_file)
    plt.title('Accuracy Evolution')
    plt.plot(data['Accuracy'], 'r-', label='Training')
    plt.legend()
    plt.savefig(image_file)
    if draws: plt.show()
    
            
        
def count_parameters(model):
    ''' Count the parameters of a model '''
    return sum(p.numel() for p in model.parameters())




