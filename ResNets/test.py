
import os
import torch 
from itertools import islice
from torch.autograd import Variable

#def avoidWarnings():
#    import warnings
#    warnings.filterwarnings('always')
#    warnings.filterwarnings('ignore')
#    warnings.filterwarnings('ignore', 'ImportWarning')
#    warnings.filterwarnings('ignore', 'DeprecationWarning')    


def test(dataset, name, singleModel, ensemble, device, dataloader, paths, save):
    
#    test = True
#    # Testing mode
#    if test:         
#        print('Validating in test mode')
#        dataloader = islice(dataloader, 20)
#    
    # Log config 
    
    logpath = paths['logs']['test']
    assert os.path.exists(logpath), 'Error: path to save test logs not found'
    logfile = name + '_test_accuracy.txt'
    logfile = os.path.join(logpath, logfile)
    if save: f = open(logfile, 'w+')
    
    
    # Single Network Performance
    
    singleModel.eval()
    singleModel.to(device)
    total, correct = 0,0
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(dataloader):
            
            images = Variable(images)
            labels = Variable(labels)
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = singleModel(images)
            
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
        
    print('Single model accuracy {}%'.format(100 * correct / total))
    if save: 
        f.write('Single model accuracy {}%'.format(100 * correct / total))
        
    
    # Ensemble Model
        
    total, correct = 0,0
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(dataloader):
            
            images = Variable(images)
            labels = Variable(labels)        
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = []
            for model in ensemble:
                
                model.eval()
                output = model(images)
                outputs.append(output)
                
            outputs = torch.mean(torch.stack(outputs), dim=0)
                
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
     
    print('Ensemble accuracy {}%'.format(100 * correct / total))
    if save: 
        f.write('Ensemble accuracy {}%'.format(100 * correct / total))
        f.close()