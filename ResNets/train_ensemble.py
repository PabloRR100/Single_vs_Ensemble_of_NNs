
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


def train(dataset, names, models, optimizers, criterion, device, trainloader, validloader,
          epochs, iters, save, paths, save_frequency=1, test=True, validate=True):
    
    # Every model train mode
    for m in models: m.train()
            
    # Initialize results
    j = 0 
    timer = []
    best_acc = 0
    results = Results(models)
    
    avoidWarnings()
#    logpath = paths['logs']['train']
    modelpath = paths['models']

    # Testing mode
    if test:         
        epochs = 3
        print('training in test mode')
        
#    # Logs config
#    if save:         
#        
#        assert os.path.exists(logpath), 'Error: path to save training logs not found'
#        logfile = names + '.txt'
#        logfile = os.path.join(logpath, logfile)
#        f = open(logfile, 'w+')

    
    start = now()
    for epoch in range(1, epochs+1):
                
        # Training
        for i, (images, labels) in enumerate(trainloader):
            
            j += 1 # for printing
            images = Variable(images)
            labels = Variable(labels)
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = []
            for n, m in enumerate(models):
                
                # Scheduler for learning rate        
                if (j == 32000 or j == 48000):  
                    for p in optimizers[m].param_groups: p['lr'] = p['lr'] / 10

                # Individual forward pass
                
                # Calculate loss for individual                
                m.zero_grad()
                output = m(images)
                outputs.append(output)
                loss = criterion(output, labels) 
                
                # Calculate accy for individual
                _, predictions = torch.max(output.data, 1)
                correct, total = 0, 0
                total += output.size(0)
                correct += int(sum(predictions == labels)) 
                accuracy = correct / total
                                
                lss = round(loss.item(), 3)
                acc = round(accuracy * 100, 2)
            
                # Store results for this individual
                results.append_loss(n, lss, 'train')
                results.append_accy(n, acc, 'train')
                
                stat = [epoch, epochs, j, iters, n]
                stats = '\n Train: Epoch: [{}/{}] Iter: [{}/{}] Model: {}%'.format(*stat)
                print(stats)      
                
                # Individual backwad pass                           # How does loss.backward wicho model is?
                loss.backward()
                optimizers[n].step()
                
            # Ensemble foward pass
            
            outputs = torch.mean(torch.stack(outputs), dim=0)
            
            # Calculate loss for ensemble
            loss = criterion(output, labels) 
            correct, total = 0, 0 
            
            # Calculate accuracy for ensemble
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
            
            lss = round(loss.item(), 3)
            acc = round(accuracy * 100, 2)
            
            # Store results for Ensemble
            results.append_loss(None, lss, 'train')
            results.append_accy(None, acc, 'train')
            
            # Print results
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
                
                outputs = []
                for n, m in enumerate(models):
            
                    m.zero_grad()
                    output = m(images)
                    outputs.append(output)
                    loss = criterion(output, labels) 
                    
                    _, predictions = torch.max(output.data, 1)
                    correct, total = 0, 0
                    total += output.size(0)
                    correct += int(sum(predictions == labels)) 
                    accuracy = correct / total
                                    
                    lss = round(loss.item(), 3)
                    acc = round(accuracy * 100, 2)
                
                    # Store results for this individual
                    results.append_loss(n, lss, 'valid')
                    results.append_accy(n, acc, 'valid')
                    
                    # Individual backwad pass                           # How does loss.backward wicho model is?
                    loss.backward()
                    optimizers[m].step()
                    
                    stat = [epoch, epochs, j, iters, n]
                    stats = '\n Train: Epoch: [{}/{}] Iter: [{}/{}] Model: {}%'.format(*stat)
                    print(stats)                    
            
            # Ensemble foward pass
            outputs = torch.mean(torch.stack(outputs), dim=0)
                
            loss = criterion(outputs, labels)  
            
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
            accuracy = correct / total
        
            lss = round(loss.item(), 3)
            acc = round(accuracy * 100, 2)
            
            # Store results for Ensemble
            results.append_loss(None, lss, 'valid')
            results.append_accy(None, acc, 'valid')
            
            # Print results
            stat = [epoch, epochs, j, iters, lss, acc]
            stats = '\n Train: Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(*stat)
            print(stats)
                
            # Save model and delete previous if it is the best
            if acc > best_acc:
                
                prev_models = glob.glob(os.path.join(modelpath, '*.pkl'))
                for o in prev_models:
                    os.remove(p)
                    
                for i, m in enumerate(models):                    
                    torch.save(m.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (names[i], epoch))) 
                best_acc = acc
                
#        if save: f.write(stats + '\n')        
        timer.append(time(start))
            
#    if save: f.close()             
    
        # DECIDE IF RETURN RESULTS OR CONVERT TO DATAFRAME HERE TO CREATE THE PLOTS
#    train_history = pd.DataFrame(np.array([train_loss, train_accy]).T, columns=['Loss', 'Accuracy'])
#    valid_history = pd.DataFrame(np.array([valid_loss, valid_accy]).T, columns=['Loss', 'Accuracy'])
#    return train_history, valid_history, timer
    return results
