

# Do not show figures
#import matplotlib
#matplotlib.use('agg') 


import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable as BT


from torchvision.transforms import transforms
from torch.utils.data.dataset import random_split
from torchvision.datasets import CIFAR10, ImageFolder


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'ImportWarning')
warnings.filterwarnings('ignore', 'DeprecationWarning')



# Errors dictionary
errors = {
        'Ensure subset': 'Choose transformations for Train set or Test set',
        'Exists data folder': 'General data folder not found',
        'Exists particular data folder': 'Not found folder for this particular dataset',
        'training' : {
                'epochs':'n_iters and batch_size must be divisible to compute the epochs',
                'batch_size':'n_iters and n_epochs must be divisible to compute the batch size'
                }
        }
        
        
# Decorator to time function executions
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed
        
        
# Count parameters of a model 
def count_parameters(model):
    ''' Count the parameters of a model '''
    return sum(p.numel() for p in model.parameters())


# EPOCHS-ITERS-BATCH_SIZE
# -----------------------

def def_training(n_iters, n_epochs, batch_size, batches, samples):
    '''
    Function to ensure the epochs, iterations and batch_sizes chosen
    are consistent
    '''
    def table_stats(n_iters, n_epochs, batch_size, batches, samples):
        table = BT()
        table.append_row(['Samples', samples])
        table.append_row(['Batch size', batch_size])
        table.append_row(['Num batches', batches])
        table.append_row(['Iters', n_iters])
        table.append_row(['Epochs', n_epochs])
        print(table)
    
    def print_stats(n_iters, n_epochs, batch_size, batches, error):
        print('\n\nERROR IN TRAINING PARAMETERS')
        print(error)
        table_stats(n_iters, n_epochs, batch_size, batches, samples)
        print('Exiting...')
    
    if n_epochs is None:
        
        n_epochs = n_iters / batches 
        error = errors['training']['epochs']
        if n_iters % batch_size != 0:
            print_stats(n_iters, n_epochs, batch_size, error)
            exit()
        n_epochs = int(n_epochs)
    
    else:
        
        # Introduced epochs and iterations
        if n_iters != 64000:
            
            print(n_iters)
            print(n_iters == 64000)
            batch_size = n_iters / n_epochs        
            error = errors['training']['batch_size']
            if n_iters % n_epochs != 0:
                print_stats(n_iters, n_epochs, batch_size, error)
                exit()            
            batch_size = int(batch_size)
          
        # Introduced epochs and batch_size
        else:
            
            n_iters = int(n_epochs * batch_size)
            
    return n_iters, n_epochs, batch_size


# DATASET 
# -------

# Define images preprocessing steps
    
def transformations(dataset, subset=None):
    
    if dataset == 'CIFAR10':
        ''' Image processing for CIFAR-10 '''
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        transformations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), padding=4), 
                transforms.ToTensor(), 
                normalize])

    if dataset == 'ImageNet':
        ''' Image processing for ImageNet '''
        assert subset is not None, errors['Ensure subset']
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        if subset == 'train':
            transformations = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])
    
        if subset == 'test':
            transformations = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])
        
    return transformations

# Load Dataset
    
def load_dataset(data_path, dataset: str, comments: bool = True):
    
    assert os.path.exists(data_path), errors['Exists data folder']
    def dataset_info(train_dataset, valid_dataset, test_dataset, name):
        
        from beautifultable import BeautifulTable as BT
        if hasattr(test_dataset, 'classes'): classes = len(test_dataset.classes)
        elif hasattr(test_dataset, 'labels'): classes = len(np.unique(test_dataset.labels))
        elif hasattr(test_dataset, 'test_labels'): classes = len(np.unique(test_dataset.test_labels))
        else: print('Classes not detected in the dataset', sys.stdout)
        
        print('Loading dataset: ', name)
        table = BT()
        table.append_row(['Train Images', len(train_dataset.indices)])
        table.append_row(['Valid Images', len(valid_dataset.indices)])
        table.append_row(['Test Images', len(test_dataset)])
        table.append_row(['Classes', classes])
        print(table)

    root = os.path.join(data_path, dataset)  
    assert os.path.exists(root), errors['Exists particular data folder']      
    
    if dataset == 'CIFAR10':
        transform = transformations(dataset)
        train_dataset = CIFAR10(root = root, download = True, train = True, transform = transform)
        test_dataset  = CIFAR10(root = root, download = False, train = False, transform = transform)
    
    if dataset == 'ImageNet':
        train_dataset = ImageFolder(root = root, transform = transformations(dataset, 'train'))
        test_dataset  = ImageFolder(root = root, transform = transformations(dataset, 'test'))
        
    if dataset == 'fruits-360-small':
        transform = transformations('CIFAR10')
        
        train_dataset = ImageFolder(root=os.path.join(root, 'Training'), transform=transform)
        test_dataset  = ImageFolder(root=os.path.join(root, 'Validation'), transform=transform)
    
    
    len_ = len(train_dataset)
    train_dataset, valid_dataset = random_split(train_dataset, [round(len_*0.9), round(len_*0.1)])
    if comments: dataset_info(train_dataset, valid_dataset, test_dataset, name=dataset)
    return train_dataset, valid_dataset, test_dataset


# Plot and save output figures
    
#import matplotlib.pyplot as plt
#def figures(data, name, dataset_name, path, draws, save):
#      
#    # Loss evolution
#    
#    image_file = 'baseline_' + name + '_' + dataset_name + '_training_loss.png'
#    image_file = os.path.join(path, image_file)
#    plt.figure()
#    plt.title('Loss Evolution')
#    plt.plot(data['Loss'], 'r-', label='Training')
#    plt.legend()
#    if save: plt.savefig(image_file)
#    if draws: plt.show()
#    
#    # Accuracy evolution
#    
#    plt.figure()
#    image_file = 'baseline_' + name + '_' + dataset_name + '_training_accuracy.png'
#    image_file = os.path.join(path, image_file)
#    plt.title('Accuracy Evolution')
#    plt.plot(data['Accuracy'], 'r-', label='Training')
#    plt.legend()
#    if save: plt.savefig(image_file)
#    if draws: plt.show()


def savefig(data: dict, path: str, title: str):
    ''' Save the plot from the data '''
    plt.figure()
    sns.set_style("dark")
    df = pd.DataFrame.from_dict(data)
    sns.lineplot(data=df)
    plt.savefig(os.path.join(path, title))
    




