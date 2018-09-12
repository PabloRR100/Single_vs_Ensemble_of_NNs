
import os
import glob
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

def train(dataset, name, model, optimizer, criterion, device, trainloader, validloader,
          epochs, iters, save, paths, save_frequency=1, test=True, validate=True):
    
    import warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', 'ImportWarning')
    warnings.filterwarnings('ignore', 'DeprecationWarning')

    best_acc = 0

    model.train()
    stats_every = 1
    logpath = paths['logs']['train']
    modelpath = paths['models']
    
    if test:         
        epochs = 3
        print('training in test mode')
    
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
            accuracy = round(correct / total * 100, 2)
            
            total_acc.append(accuracy)
            total_loss.append(round(loss.item(), 3))
            
            stats = 'Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(
                        epoch, epochs, j, iters, round(loss.item(), 2), accuracy)
            
        # Validation
        if validate:
            
            print('Entering validation')
            for k, (images, labels) in enumerate(validloader):
            
                images = Variable(images)
                labels = Variable(labels)
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                _, preds = outputs.max(1)
                total += outputs.size(0)
                correct += int(sum(preds == labels))
                acc = 100 * correct / total
                
                if acc > best_acc:
                    models = glob.glob(modelpath)
                    for m in models:
                        os.remove(m)
                    torch.save(model.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (name, epoch))) 
                    best_acc = acc

            
            if j % stats_every == 0: print('\n' + stats)
            if save: f.write(stats + '\n')
        
        total_time.append(time(start))
            
    if save: f.close()             
    
    train_history = pd.DataFrame(np.array([total_loss, total_acc]).T, columns=['Loss', 'Accuracy'])
    return train_history, total_time
