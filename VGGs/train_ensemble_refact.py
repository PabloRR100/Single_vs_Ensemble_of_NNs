
import os
import glob
import torch
from utils import timeit
from datetime import datetime 
from torch.autograd import Variable
from results import TrainResults as Results


def avoidWarnings():
    import warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', 'ImportWarning')
    warnings.filterwarnings('ignore', 'DeprecationWarning')   
    
    
now = datetime.now
def elapsed(start):
    return round((now() - start).seconds/60, 2)


def time():
    print('{}:{} \n'.format(now().hour, now().minute))


def print_stats(epoch, epochs, j, iters, lss, acc, subset, n=None):
    if n:
        stat = [subset, n, epoch, epochs, j, iters, lss, acc]        
        stats = '\n {} Model {}: Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(*stat)
    else:
        stat = [subset, epoch, epochs, j, iters, lss, acc]        
        stats = '\n {} Ensemble: Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(*stat)
    print(stats)    
    

@timeit
def train(dataset, names, models, optimizers, criterion, device, trainloader, validloader,
          epochs, iters, save, paths, test=True, validate=True):
    
    com_iter = False
    com_epoch = True
    # Every model train mode
    for m in models: m.train()
            
    # Initialize results
    j = 0 
    best_acc = 0
    results = Results(models)
    len_ = len(trainloader)
    
    avoidWarnings()
    modelpath = paths['models']

    start = now()
    results.append_time(0)
    results.name = names[0][:-2] + '(x' + str(len(names)) + ')'
    
    
    def train_epoch():
        global j, results
        
        for i, (images, labels) in enumerate(trainloader):
            
            j += 1 # for printing
            images, labels = Variable(images), Variable(labels)            
            images, labels = images.to(device), labels.to(device)            
            outs = train_minibatch(images, labels)
            
            ## Ensemble foward pass
            output = torch.mean(torch.stack(outs), dim=0)
            
            # Calculate loss for ensemble
            loss = criterion(output, labels) 
            correct, total = 0, 0 
            
            # Calculate accuracy for ensemble
            _, preds = output.max(1)
            total += output.size(0)
            correct += int(sum(preds == labels))
            accuracy = correct / total
            
            lss = round(loss.item(), 3)
            acc = round(accuracy * 100, 2)
            
            # Store iteration results for Ensemble
            results.append_iter_loss(lss, 'train', None)
            results.append_iter_accy(acc, 'train', None)
            
            # Print results
            if com_iter: print_stats(epoch, epochs, j, iters, lss, acc, 'Train')
    
    def train_minibatch(images, labels):
        global models, criterion, optimizers, results
        
        outs = []
        for n, m in enumerate(models):
            
            # Scheduler for learning rate        
            if (j == 32000 or j == 48000):  
                for p in optimizers[n].param_groups: p['lr'] = p['lr'] / 10

            ## Individual forward pass
            
            # Calculate loss for individual                
            m.zero_grad()
            output = m(images)
            outs.append(output)
            loss = criterion(output, labels) 
            
            # Calculate accy for individual
            _, predictions = torch.max(output.data, 1)
            correct, total = 0, 0
            total += output.size(0)
            correct += int(sum(predictions == labels)) 
            accuracy = correct / total
                            
            lss = round(loss.item(), 3)
            acc = round(accuracy * 100, 2)
        
            # Store iteration results for this individual
            results.append_iter_loss(lss, 'train', n+1)
            results.append_iter_accy(acc, 'train', n+1)
            
            if i == len_-1:
                # Store epoch results for this individual (as last iter)
                results.append_loss(lss, 'train', n+1)
                results.append_accy(acc, 'train', n+1)
                
            if com_iter: print_stats(epoch, epochs, j, iters, lss, acc, 'Train', n+1)  
            
            # Individual backwad pass                           # How does loss.backward wicho model is?
            
            loss.backward()
            optimizers[n].step()        
        return outs
        
    def validate_epoch(validloader):
        
        correct, total = 0, 0
        for k, (images, labels) in enumerate(validloader):
        
            images, labels = Variable(images), Variable(labels)            
            images, labels = images.to(device), labels.to(device)    
            output = validate_minibatch(images, labels)
            
        
        loss = criterion(output, labels) 
        accuracy = correct / total    
        lss = round(loss.item(), 3)
        acc = round(accuracy * 100, 2)
        
        # Store epoch results for Ensemble
        results.append_loss(lss, 'valid', None)
        results.append_accy(acc, 'valid', None)
        
        # Print results
        if com_epoch: print_stats(epoch, epochs, j, iters, lss, acc, 'Valid', None)
            
        # Save model and delete previous if it is the best
        if acc > best_acc:
            
            prev_models = glob.glob(os.path.join(modelpath, names[0][:-2] + '*.pkl'))
            for p in prev_models:
                os.remove(p)
                
            for i, m in enumerate(models):                    
                torch.save(m.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (names[i], epoch))) 
            best_acc = acc
    
    def validate_minibatch(images, labels):
        outs = []
        for n, m in enumerate(models):
            
            ## Individuals foward pass
    
            m.zero_grad()
            output = m(images)
            outs.append(output)
                                
            # Store epoch results for each model
            if k == 0:
                
                loss = criterion(output, labels) 
            
                corr, tot = 0, 0
                _, predictions = torch.max(output.data, 1)
                tot += output.size(0)
                corr += int(sum(predictions == labels)) 
                accur = corr / tot
                                
                ls = round(loss.item(), 3)
                ac = round(accur * 100, 2)
            
                results.append_loss(ls, 'valid', n+1)
                results.append_accy(ac, 'valid', n+1)
                
                if com_epoch: 
                    print_stats(epoch, epochs, j, iters, ls, ac, 'Valid', n+1)
            
        ## Ensemble foward pass
        output = torch.mean(torch.stack(outs), dim=0)
                            
        _, preds = output.max(1)
        total += output.size(0)
        correct += int(sum(preds == labels))
        return out
        
    
    
    for epoch in range(1, epochs+1):
        
        if epoch % 10 == 0: time()
        
        # Train ensemble for an epoch
        lss, acc = train_epoch(trainloader)
        # Store epoch results for Ensemble
        results.append_loss(lss, 'train', None)
        results.append_accy(acc, 'train', None)
                
        # Print results
        if com_epoch: print_stats(epoch, epochs, j, iters, lss, acc, 'Train')
            
        # Validation
        # ----------
        if validate:
            
            
            
        results.append_time(elapsed(start))
        
    return results