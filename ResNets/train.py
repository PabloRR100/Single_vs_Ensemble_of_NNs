
import sys
sys.path.append('..')
from datetime import datetime 
from utils import progress_bar
from torch.autograd import Variable

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore', 'ImportWarning')
warnings.filterwarnings('ignore', 'DeprecationWarning')


now = datetime.now
def time(start):
    
    elapsed = (now() - start).total_seconds()
    hours =  int(elapsed/3600)
    minutes = round((elapsed/3600 - hours)*60, 2)
    return hours, minutes


def train(model, optimizer, criterion, device, dataloader, epochs, iters):
    
    model.train()
    
    j = 0 
    train_loss = 0
    total_loss = []
    total_acc = []
    
    for epoch in range(epochs):
        
        # Scheduler for learning rate        
        if (j == 32000 or j == 48000):
            for p in optimizer.param_groups:  p['lr'] = p['lr'] / 10
        
        for i, (images, labels) in enumerate(dataloader):
            
            j += 1
            
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
        