

import os
import glob
import torch
import numpy as np
import pandas as pd
from results import Results
from datetime import datetime 
from torch.autograd import Variable


def avoidWarnings():
    import warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', 'ImportWarning')
    warnings.filterwarnings('ignore', 'DeprecationWarning')    

now = datetime.now
def time(start):
    ''' Helper function to track time wrt an anchor'''
    elapsed = (now() - start).total_seconds()
    hours =  int(elapsed/3600)
    minutes = round((elapsed/3600 - hours)*60, 2)
    return hours, minutes



def train(dataset, name, model, optimizer, criterion, device, trainloader, validloader,
          epochs, iters, save, paths, save_frequency=1, test=True, validate=True):
    
    best_acc = 0
    model.train()
    results = Results([model])
    
    avoidWarnings()
    modelpath = paths['models']
    
    # Testing mode
    if test:         
        epochs = 10
        print('training in test mode')
    
    timer = []
    j = 0 # Iteration controler
    
    start = now()
    for epoch in range(1, epochs+1):
        
        # Scheduler for learning rate        
        if (j == 32000 or j == 48000):  
            for p in optimizer.param_groups: p['lr'] = p['lr'] / 10
        
        # Training
        for i, (images, labels) in enumerate(trainloader):
            
            j += 1 # for printing
            images = Variable(images)
            labels = Variable(labels)
            
            images = images.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            outputs = model(images)
            scores, predictions = torch.max(outputs.data, 1)
        
            loss = criterion(outputs, labels)            
            loss.backward()
            optimizer.step()

            correct, total = 0, 0
            total += outputs.size(0)
            correct += int(sum(predictions == labels)) 
            accuracy = correct / total
            
            lss = round(loss.item(), 3)
            acc = round(accuracy * 100, 2)
            
            # Stores per iteration results
            results.append_global_loss(lss, 'train')
            results.append_global_accy(acc, 'train')
          
        # Stores per-epoch results
        results.append_loss(lss, 'train')
        results.append_accy(acc, 'train')
        
        stat = [epoch, epochs, j, iters, lss, acc]
        stats = '\n Train: Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(*stat)
        print(stats)
        
        # Validation
        if validate:
            
            correct, total = 0, 0
            for k, (images, labels) in enumerate(validloader):
            
                images = Variable(images)
                labels = Variable(labels)
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                loss = criterion(outputs, labels)  
                
                _, preds = outputs.max(1)
                total += outputs.size(0)
                correct += int(sum(preds == labels))
                accuracy = correct / total
            
                lss = round(loss.item(), 3)
                acc = round(accuracy * 100, 2)
                
            # Save model and delete previous if it is the best
            if acc > best_acc:
                
                models = glob.glob(os.path.join(modelpath, '*.pkl'))
                for m in models:
                    os.remove(m)
                torch.save(model.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (name, epoch))) 
                best_acc = acc
        
            results.append_loss(lss, 'valid')
            results.append_accy(acc, 'valid')
            
            stat = [epoch, epochs, j, iters, lss, acc]
            stats = '\n Valid: Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(*stat)
            print(stats)
            
        timer.append(time(start))
        
    return results, timer