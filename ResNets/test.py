
import os
import torch 
from torch.autograd import Variable


def test(dataset, singleModel, ensemble, test_loader, logpath, save):
    
    print('\n Calculating test accuracy... \n')
    # Log config 
    
    assert os.path.exists(logpath), 'Error: path to save test logs not found'
    logfile = 'test_' + singleModel.name + '.txt'
    logfile = os.path.join(logpath, logfile)
    if save: f = open(logfile, 'w+')
    
    
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
            
    if save: 
        print('Single model accuracy {}%'.format(100 * correct / total))
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
     
    if save: 
        print('Ensemble accuracy {}%'.format(100 * correct / total))
        f.write('Ensemble accuracy {}%'.format(100 * correct / total))
        f.close()
    

           
        
        
        
        
        
        
        
''' 
Backup code

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
        
        if (i % 50 == 0 and (i < 100 or (i > 1000 and i < 1050))) or i % 1000 == 0:
            print('Image [{}/{}]. Total correct {}'.format(i,len(test_set),correct))                
    
    print('Accuracy of the network on the {} test images: {}%'.format(len(test_set), (100 * correct / total)))
     
        
# Ensemble Model
    
total, correct = 0,0
with torch.no_grad():
    
    for i, (images, labels) in enumerate(test_loader):
        
        #images, labels = test_set[0]
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
        
        if (i % 50 == 0 and (i < 100 or (i > 1000 and i < 1050))) or i % 1000 == 0:
            print('Image [{}/{}]. Total correct {}'.format(i,len(test_set),correct))                
            
    print('Accuracy of the network on the {} test images: {}%'.format(len(test_set), (100 * correct / total)))
'''