# -*- coding: utf-8 -*-


import os
import torch 
from torch.autograd import Variable


def test(dataset, singleModel, ensemble, test_loader, logpath):
    
    # Log config 
    
    assert os.path.exists(logpath), 'Error: path to save train logs not found'
    logfile = 'test_' + singleModel.name + '.txt'
    logfile = os.path.join(logpath, logfile)
    f = open(logfile)
    
    
    # Single Network Performance
    
    singleModel.eval()
    total, correct = 0,0
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(test_loader):
            
            images = Variable(images)
            labels = Variable(labels)
            
            outputs = singleModel(images)
            
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
            
    f.write('Single model accuracy {}%'.format(100 * correct / total))
    f.close()
        
    
    # Ensemble Model
        
    total, correct = 0,0
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(test_loader):
            
            images = Variable(images)
            labels = Variable(torch.tensor(labels))        
            
            outputs = []
            for model in ensemble:
                
                model.eval()
                output = model(images)
                outputs.append(output)
                
            outputs = torch.mean(torch.stack(outputs), dim=0)
                
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
            
    f.write('Single model accuracy {}%'.format(100 * correct / total))
    f.close()
           