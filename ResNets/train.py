#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:35:14 2018
@author: pabloruizruiz
"""

import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime 
from torch.autograd import Variable


now = datetime.now
def time(start):
    elapsed = (now() - start).total_seconds()
    hours =  int(elapsed/3600)
    minutes = round((elapsed/3600 - hours)*60, 2)
    return hours, minutes


def train(dataset, model, optimizer, criterion, trainloader, epochs, iters, 
          createlog=False, logpath=None, print_every=2, save_frequency=1):
    
    debugging = True
    
    if createlog:        
        logfile = model.name + '.txt'
        logfile = os.path.join(logpath, logfile)
        f = open(logfile)
    
    j = 0           # Iteration controler  
    total_time = []
    total_loss = []
    total_acc = []
    
    start = now()
    for epoch in range(epochs):
    
        # Scheduler for learning rate        
        if (dataset == 'CIFAR10' and (j == 32000 or j == 48000)) or \
            (dataset == 'ImageNet' and (epoch == 30 or epoch == 60)):  
            ## TODO: change this to match paper, change lr when error plateas
                
            for p in optimizer.param_groups: p['lr'] = p['lr'] / 10
            for p in optimizer.param_groups: p['lr'] = p['lr'] / 10
        
        for i, (images, labels) in enumerate(trainloader):
            
            i += 1; j += 1
            images = Variable(images)
            labels = Variable(labels)
            
            model.zero_grad()
            outputs = model(images)
            scores, predictions = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)            
            loss.backward()
            optimizer.step()
            
            correct, total = 0, 0
            total += outputs.size(0)
            correct += int(sum(predictions == labels)) 
            accuracy = round(correct / total, 3)
            
            total_acc.append(accuracy)
            total_loss.append(round(loss.item(), 3))
            
            if j % print_every == 0:
                
                stats = 'Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}'.format(
                        epoch, epochs, j, iters, round(loss.item(), 2), accuracy)
                
                print('\n' + stats)
                if createlog: f.write(stats + '\n')
                
            if debugging and i > 3: break # To track training

        if debugging and epoch > 4: break
        total_time.append(time(start))        
        if save_frequency is not None and epoch % save_frequency == 0:
            torch.save(model.state_dict(), os.path.join('./models', '%s-%d.pkl' % (model.name, epoch))) 
            
    print('Epoch: {} Time: {} hours {} minutes'.format(epoch, time(start)[0], time(start)[1]))
    
    if createlog: f.close()             

    train_history = pd.DataFrame(np.array([total_loss, total_acc]).T, columns=['Loss', 'Accuracy'])
    return train_history, total_time


