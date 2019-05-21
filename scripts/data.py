#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:16:53 2019
@author: pabloruizruiz
"""

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def create_data_loaders(batch_size, workers):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../datasets/CIFAR10/', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../datasets/CIFAR10/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    return train_loader, test_loader



def create_data_loaders_SVHN(batch_size, workers):
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root='../../datasets/SVHN/', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root='../../datasets/SVHN/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    return train_loader, test_loader
