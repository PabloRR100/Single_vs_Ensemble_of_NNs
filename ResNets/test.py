
import torch 
from torch.autograd import Variable


def test(dataset, name, singleModel, ensemble, device, dataloader, best_acc):
    
    # Single Network Performance    
    total, correct = 0,0    
    singleModel.eval()
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(dataloader):
            
            images = Variable(images)
            labels = Variable(labels)
            
            outputs = singleModel(images)
            
            _, preds = outputs.max(1)
            total += outputs.size(0)
            correct += int(sum(preds == labels))
            
    print('Single model accuracy {}%'.format(100 * correct / total))
        
    
    # Ensemble Model
        
    total, correct = 0,0
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(dataloader):
            
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
     
    print('Ensemble accuracy {}%'.format(100 * correct / total))    
          