
import os
import glob
import torch
from datetime import datetime 
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from metrics import accuracies, AverageMeter
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
    print('\n ** Time {}:{}'.format(now().hour, now().minute))


def print_stats(epoch, epochs, j, iters, lss, acc, subset):
    
    if subset == 'Train': time()
    stat = [subset, epoch, epochs, j, iters, lss, acc]        
    stats = '\n ** {} ** Epoch: [{}/{}] Iter: [{}/{}] Loss: {} Acc: {}%'.format(*stat)
    print(stats)    
    


def train(name, model, optimizer, criterion, device, trainloader, validloader, epochs, paths):
    
    j = 0 
    best_acc = 0
    com_iter = False
    com_epoch = True
    results = Results([model])
    iters = epochs * len(trainloader)
    
#    top1 = AverageMeter()
#    top5 = AverageMeter()
#    losses = AverageMeter()
    
    avoidWarnings()
    modelpath = paths['models']
        
    start = now()
    results.name = name
    results.timer.append(0)
    
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    
    cudnn.benchmark = True
    print('\nStarting Single Model Training...' )
    for epoch in range(1, epochs+1):
        
        done = lambda x,y: round(x/y,2)
        if (done(epoch, epochs) == 0.50 or done(epoch, epochs) == 0.75):
            for p in optimizer.param_groups: 
                p['lr'] = p['lr'] / 10
            print('\n** Changing LR to {} \n'.format(p['lr']))
        
        # Training
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            
            j += 1 # for printing
            images = Variable(images)
            labels = Variable(labels)
            
            images = images.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)            
        
            # measure accuracy and record loss
#            prec1, prec5 = accuracies(outputs.data, labels.data, topk=(1, 5))
#            losses.update(loss.data[0], images.size(0))
#            top1.update(prec1[0], images.size(0))
#            top5.update(prec5[0], images.size(0))
            scores, predictions = torch.max(outputs.data, 1)

            # Compute gradient and do SGD step
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
#            results.append_iter_accy((prec1, prec5), 'train')
          
            if com_iter: print_stats(epoch, epochs, j, iters, lss, acc, 'Train')  
            
        # Stores per-epoch results
        results.append_loss(lss, 'train')
        results.append_accy(acc, 'train')
#        results.append_accy((prec1, prec5), 'train')
        
        if com_epoch: print_stats(epoch, epochs, j, iters, lss, acc, 'Train')  
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            
            for k, (images, labels) in enumerate(validloader):
            
                images = Variable(images)
                labels = Variable(labels)
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                loss = criterion(outputs, labels)  
                
                 # measure accuracy and record loss
#                prec1, prec5 = accuracies(outputs.data, labels.data, topk=(1, 5))
#                losses.update(loss.data[0], images.size(0))
#                top1.update(prec1[0], images.size(0))
#                top5.update(prec5[0], images.size(0))
                _, preds = outputs.max(1)
                total += outputs.size(0)
                correct += int(sum(preds == labels))
            
        accuracy = correct / total
    
        lss = round(loss.item(), 3)
        acc = round(accuracy * 100, 2)
        
        if com_epoch: print_stats(epoch, epochs, j, iters, lss, acc, 'Valid')

        # Save model and delete previous if it is the best
        if acc > best_acc:
            
            # print('Best validation accuracy reached --> Saving model')              
            models = glob.glob(os.path.join(modelpath, '*.pkl'))
            for m in models:
                os.remove(m)
            torch.save(model.state_dict(), os.path.join(modelpath, '%s-%d.pkl' % (name, epoch))) 
            best_acc = acc
    
            # Store per-epoch results
            results.append_loss(lss, 'valid')
            results.append_accy(acc, 'valid')
#            results.append_accy((prec1, prec5), 'valid')
        
        results.append_time(elapsed(start))
        
    print('\nFinished training... Time: ', elapsed(start))
    return results
