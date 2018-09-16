
import os
import glob
import torch
from results import Results
from itertools import islice
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


def print_stats(epoch, epochs, j, iters, lss, acc, subset):
    
    stat = [subset, epoch, epochs, j, iters, lss, acc]        
    stats = '\n {}:: Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(*stat)
    print(stats)    
    


def train(dataset, name, model, optimizer, criterion, device, trainloader, validloader,
          epochs, iters, save, paths, save_frequency=1, test=True, validate=True):
    
    j = 0 
#    timer = []
    best_acc = 0
    model.train()
    com_iter = False
    com_epoch = True
    results = Results([model])
    
    avoidWarnings()
    modelpath = paths['models']
    
    # Testing mode
    if test:         
        epochs = 2
#        print('training in test mode')
#        trainloader = islice(trainloader, 2)
#        validloader = islice(validloader, 2)
    
    start = now()
    results.timer.append(0)
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
            results.append_iter_loss(lss, 'train')
            results.append_iter_accy(acc, 'train')
          
            if com_iter: print_stats(epoch, epochs, j, iters, lss, acc, 'Train')  
            
        # Stores per-epoch results
        results.append_loss(lss, 'train')
        results.append_accy(acc, 'train')
        
        if com_epoch: print_stats(epoch, epochs, j, iters, lss, acc, 'Train')  
        
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
            
#            # Save model and delete previous if it is the best
#            if acc > best_acc:
#                
#                models = glob.glob(os.path.join(modelpath, '*.pkl'))
#                for m in models:
#                    os.remove(m)
#                torch.save(model.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (name, epoch))) 
#                best_acc = acc
        
            # Store per-epoch results
            results.append_loss(lss, 'valid')
            results.append_accy(acc, 'valid')
            
            if com_epoch: print_stats(epoch, epochs, j, iters, lss, acc, 'Valid')              
        
        results.append_time(time(start))
#        timer.append(time(start))
        
#    return results, timer
    return results