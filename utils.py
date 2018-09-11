
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data.dataset import random_split
from torchvision.datasets import CIFAR10, ImageFolder


# Errors dictionary
errors = {
        'Ensure subset': 'Choose transformations for Train set or Test set',
        'Exists data folder': 'General data folder not found',
        'Exists particular data folder': 'Not found folder for this particular dataset',
        }


# Count parameters of a model 

def count_parameters(model):
    ''' Count the parameters of a model '''
    return sum(p.numel() for p in model.parameters())

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
    
def figures(data, name, dataset_name, path, draws, save):
      
    # Loss evolution
    
    image_file = 'baseline_' + name + '_' + dataset_name + '_training_loss.png'
    image_file = os.path.join(path, image_file)
    plt.figure()
    plt.title('Loss Evolution')
    plt.plot(data['Loss'], 'r-', label='Training')
    plt.legend()
    if save: plt.savefig(image_file)
    if draws: plt.show()
    
    # Accuracy evolution
    
    plt.figure()
    image_file = 'baseline_' + name + '_' + dataset_name + '_training_accuracy.png'
    image_file = os.path.join(path, image_file)
    plt.title('Accuracy Evolution')
    plt.plot(data['Accuracy'], 'r-', label='Training')
    plt.legend()
    if save: plt.savefig(image_file)
    if draws: plt.show()
    
            
    
# MONITOR PERFORMANCE
# -------------------
    
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f




