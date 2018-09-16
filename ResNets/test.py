
import torch 
from torch.autograd import Variable

def test(dataset, name, singleModel, ensemble, device, dataloader, paths, save):
            
    # Single Network Performance
    
    singleModel.eval()
    for m in ensemble: m.eval()
    
    control = 0

    total, correct = (0,0)
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
            
            control += 1
        
    print('Single model accuracy {}%'.format(100 * correct / total))
    print('Control: ', control)
        
    
    # Ensemble Model 
    
    control = 0    
    
    total, correct = (0,0)
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(dataloader):
            
            images = Variable(images)
            labels = Variable(labels)        
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = []
            for model in ensemble:
                
                output = model(images)
                outputs.append(output)
                
            outputs = torch.mean(torch.stack(outputs), dim=0)
                
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
     
    print('Ensemble accuracy {}%'.format(100 * correct / total))
    print('Control: ', control)