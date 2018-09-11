
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime 
from torch.autograd import Variable

import sys
sys.path.append('..')
from utils import progress_bar

import warnings
warnings.filterwarnings('ignore', 'always')

now = datetime.now
def time(start):
    
    elapsed = (now() - start).total_seconds()
    hours =  int(elapsed/3600)
    minutes = round((elapsed/3600 - hours)*60, 2)
    return hours, minutes


def train(dataset, name, model, optimizer, criterion, device, dataloader, 
          epochs, iters, save, paths, save_frequency=1, test=True):
    
    model.train()
#    logpath = paths['logs']['train']
#    modelpath = paths['models']
#    
#    # test: reduce the training for testing purporse
#    if test: 
#        
#        epochs = 5
#        print('training in test mode')
#        # dataloader = islice(dataloader, 2)
#    
#    # Logs config
#    if save:        
#        
#        assert os.path.exists(logpath), 'Error: path to save training logs not found'
#        logfile = name + '.txt'
#        logfile = os.path.join(logpath, logfile)
#        f = open(logfile, 'w+')
    
    j = 0 
    train_loss = 0
    total_loss = []
    total_acc = []
    
    for epoch in range(epochs):
        
        # Scheduler for learning rate        
        if (j == 32000 or j == 48000):
            for p in optimizer.param_groups:  p['lr'] = p['lr'] / 10
        
        for i, (images, labels) in enumerate(dataloader):
            
            j += 1 # for printing
            images = Variable(images)
            labels = Variable(labels)
            
            images = images.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            outputs = model(images)
        
            loss = criterion(outputs, labels)            
            loss.backward()
            optimizer.step()

            correct, total = 0, 0
            total += outputs.size(0)
            _, predictions = outputs.max(1)
            correct += predictions.eq(labels).sum().item()
            accuracy = round(correct / total * 100, 2)
    
            train_loss += loss.item()
            total_acc.append(accuracy)
            total_loss.append(round(loss.item(), 3))
            
            progress_bar(i, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(i+1), accuracy, correct, total))
        
#        if save and (save_frequency is not None and epoch % save_frequency == 0):
#            torch.save(model.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (name, epoch))) 
#
#    if save: f.close()             
#
#    train_history = pd.DataFrame(np.array([total_loss, total_acc]).T, columns=['Loss', 'Accuracy'])
#    return train_history, total_time
