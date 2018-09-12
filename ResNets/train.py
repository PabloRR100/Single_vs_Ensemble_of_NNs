
import os
import torch
import numpy as np
import pandas as pd
from itertools import islice
from datetime import datetime 
from torch.autograd import Variable


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'ImportWarning')
warnings.filterwarnings('ignore', 'DeprecationWarning')


now = datetime.now
def time(start):
    
    elapsed = (now() - start).total_seconds()
    hours =  int(elapsed/3600)
    minutes = round((elapsed/3600 - hours)*60, 2)
    return hours, minutes

def train(dataset, name, model, optimizer, criterion, device, dataloader, 
          epochs, iters, save, paths, save_frequency=1, test=True):
    
    model.train()
    stats_every = 1
    logpath = paths['logs']['train']
    modelpath = paths['models']
    
    # test: reduce the training for testing purporse
    if test: 
        
        epochs = 1
        print('training in test mode')
        # dataloader = islice(dataloader, 2)
    
    # Logs config
    if save:        
        
        assert os.path.exists(logpath), 'Error: path to save training logs not found'
        logfile = name + '.txt'
        logfile = os.path.join(logpath, logfile)
        f = open(logfile, 'w+')
    
    j = 0 # Iteration controler  
    total_time = []
    total_loss = []
    total_acc = []
    
    start = now()
    for epoch in range(epochs):
        
        print('Epoch ', epoch)
#        estart = now()
        # Scheduler for learning rate        
        if (dataset == 'CIFAR10' and (j == 32000 or j == 48000)) or \
            (dataset == 'ImageNet' and (epoch == 30 or epoch == 60)):  
            ## TODO: change this to match paper, change lr when error plateas
                
            print('Learning rate escheduler')
            for p in optimizer.param_groups: 
                prev_lr = p['lr']
                p['lr'] = prev_lr / 10
                print('Previous lr = {}. New lr = {}'.format(prev_lr, p['lr']))
            for p in optimizer.param_groups:
                prev_lr = p['lr']
                p['lr'] = prev_lr / 10
                print('Previous lr = {}. New lr = {}'.format(prev_lr, p['lr']))
        
        for i, (images, labels) in enumerate(dataloader):
            
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
            accuracy = round(correct / total * 100, 2)
            
            total_acc.append(accuracy)
            total_loss.append(round(loss.item(), 3))
            
            stats = 'Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(
                        epoch, epochs, j, iters, round(loss.item(), 2), accuracy)
            
            if j % stats_every == 0: print('\n' + stats)
            if save: f.write(stats + '\n')
        
        total_time.append(time(start))
#        print('Epoch time: {} hours {} minutes'.format(time(start)[0], time(start)[1]))
#        print('Total time: {} hours {} minutes'.format(time(estart)[0], time(estart)[1]))
        
        if save and (save_frequency is not None and epoch % save_frequency == 0):
            torch.save(model.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (name, epoch))) 

    if save: f.close()             

    train_history = pd.DataFrame(np.array([total_loss, total_acc]).T, columns=['Loss', 'Accuracy'])
    return train_history, total_time
