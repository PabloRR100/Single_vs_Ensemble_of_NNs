 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:31:56 2018
@author: pabloruizruiz
@title: Deep ResNets vs Shallow ResNets Ensemble on CIFAR-10
"""

import os
import math
import pickle
from beautifultable import BeautifulTable as BT

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('..')
from utils import count_parameters

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'ImportWarning')
warnings.filterwarnings('ignore', 'DeprecationWarning')


''' 
CONFIGURATION 
-------------
'''
print('\n\nCONFIGURATION')
print('-------------')


comments = True                         # Log erbosity
dataset = 'CIFAR10'                     # Choose dataset
model = 'Ensemble_Non_Recursive'        # Choose architecture
    
if model == 'Ensemble_Non_Recursive':
    n_epochs = 500
    batch_size = 128
    milestones = [150, 300, 400]

momentum = 0.9
learning_rate = 0.1
weight_decay = 1e-4
load_trained_models = False

cuda = torch.cuda.is_available()
n_workers = torch.multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = True if torch.cuda.device_count() > 1 else False
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPUs'
mem = False if device == 'cpu' else True


table = BT()
table.append_row(['Python Version', sys.version[:5]])
table.append_row(['PyTorch Version', torch.__version__])
table.append_row(['Device', str(device_name)])
table.append_row(['Cores', str(n_workers)])
table.append_row(['GPUs', str(torch.cuda.device_count())])
table.append_row(['CUDNN Enabled', str(torch.backends.cudnn.enabled)])
table.append_row(['Architecture', model])
table.append_row(['Dataset', dataset])
table.append_row(['Epochs', str(n_epochs)])
table.append_row(['Batch Size', str(batch_size)])

print(table)




'''
DEFININTION OF PATHS 
--------------------
Define all the paths to load / save files
Ensure all those paths are correctly defined before moving on
'''

#folder = 'recursives/single_non_recursive'
#folder = 'recursives/single_recursive'
folder = 'recursives/ensemble_non_recursives'
#folder = 'recursives/ensemble_recursives'

print('\n\nDEFINITION OF PATHS')
print('-------------------')
scripts = os.getcwd()
root = os.path.abspath(os.path.join(scripts, '../'))
results = os.path.abspath(os.path.join(root, 'results'))
data_path = os.path.abspath(os.path.join(root, '../datasets'))

path_to_logs = os.path.join(results, 'logs', folder)
path_to_models = os.path.join(results, 'models', folder)
path_to_figures = os.path.join(results, 'figures', folder)
path_to_definitives = os.path.join(path_to_models, 'definitives')

assert os.path.exists(root), 'Root folder not found: {}'.format(root)
assert os.path.exists(scripts), 'Scripts folder not found: {}'.format(scripts)
assert os.path.exists(results), 'Results folder not found: {}'.format(results)
assert os.path.exists(data_path), 'Data folder not found: {}'.format(data_path)
assert os.path.exists(path_to_logs), 'Logs folder not found: {}'.format(path_to_logs)
assert os.path.exists(path_to_models), 'Models folder not found: {}'.format(path_to_models)
assert os.path.exists(path_to_figures), 'Figure folder not found: {}'.format(path_to_figures)
assert os.path.exists(path_to_definitives), 'Def. models folder not found: {}'.format(path_to_definitives)

print('\n[OK]: Paths Validated Successfully')
print('Root path: ', root)
print('Script path: ', scripts)
print('Results path: ', results)
print('DataFolder path: ', data_path)
print('Models to save path: ', path_to_models)
print('Models to load path: ', path_to_definitives)

paths = {
    'root': root, 
    'script': scripts,
    'data': data_path,
    'results': results,
    'logs': path_to_logs, 
    'models': path_to_models,
    'definitives': path_to_definitives,
    'figures': path_to_figures
}



# 1 - Import the Dataset
# ----------------------

print('\n\nIMPORTING DATA')
print('--------------')

from data import create_data_loaders
train_loader, valid_loader = create_data_loaders(batch_size, n_workers)


# 2 - Import the Models
# ---------------------

print('\n\nLOADING MODELS')
print('----------------')

from collections import OrderedDict

E = 3
L = 16
M = 32
    
if model == 'Single_Non_Recursive':

    from models import Conv_Net
    net = Conv_Net('net', layers=L, filters=M, normalize=False)
    
    print('Regular net')
    if comments: print(net)
    print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))


elif model == 'Single_Recursive':
    
    from models import Conv_Recusive_Net
    net = Conv_Recusive_Net('recursive_net', L, M)
    
    print('Recursive ConvNet')
    if comments: print(net)
    print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))
        

elif model == 'Ensemble_Non_Recursive':
    
    from models import Conv_Net    
    net = Conv_Net('Convnet', L, M)
    
    print('Non Recursive ConvNet')
    if comments: print(net)
    print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))
        
    ensemble = OrderedDict()
    for n in range(1,1+E):
        ensemble['net_{}'.format(n)] = Conv_Net('net_{}'.format(n), L, M)

elif model == 'Ensemble_Non_Recursive':
    
    from models import Conv_Recusive_Net
    net = Conv_Recusive_Net('recursive_net', L, M)

    print('Recursive ConvNet')
    if comments: print(net)
    print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))

    ensemble = OrderedDict()
    for n in range(1,1+E):
        ensemble['net_{}'.format(n)] = Conv_Recusive_Net('net_{}'.format(n), L, M)    

else:
    print('Model chosen not valid')

# Apply constraint - Parameters constant

singleModel = Conv_Net('Convnet', L, M) 
name = title = singleModel.name
optimizer = optim.SGD(singleModel.parameters(), learning_rate, momentum, weight_decay)


# Construct the ensemble

names = []
ensemble = []
optimizers = []
for i in range(E):
    
    model = Conv_Net('Convnet', L, M) 
    ensemble.append(model)
    names.append(model.name + '_' + str(i+1))
    params = optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    optimizers.append(params)


#print('Existing - just for debuging on the cloud purpose')
#exit()


# 3 - Train ResNet
# ----------------

print('\n\nTRAINING')
print('--------')

criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()


## Single Model
#
#from train import train
#n_iters = n_epochs * len(train_loader)
#params = [name, singleModel, optimizer, criterion, device, train_loader, valid_loader, n_epochs, paths]
#
#results = train(*params)
#with open(name + '_Results_Single_Models.pkl', 'wb') as object_result:
#    pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL)
#
#results.show()


# Ensemble Model

n_epochs = 5
from train_ensemble import train as train_ensemble
params = [names, ensemble, optimizers, criterion, device, train_loader, valid_loader, n_epochs, paths]
    
# Start Training
import click
print('Current set up')
print('[ALERT]: Path to results (this may overwrite', paths['results'])
print('[ALERT]: Path to checkpoint (this may overwrite', None)
if click.confirm('Do you want to continue?', default=True):

    print('[OK]: Starting Training of Recursive Ensemble Model')
    ens_results = train_ensemble(*params)
    with open(name + '_Results_Ensemble_Models.pkl', 'wb') as object_result:
        pickle.dump(ens_results, object_result, pickle.HIGHEST_PROTOCOL)
    
ens_results.show()


# 4 - Evaluate Models
# -------------------
    
print('\n\nTESTING')
print('-------')

from test import test

testresults = test(dataset, name, singleModel, ensemble, device, valid_loader, paths)
with open(name + '_Results_Testing.pkl', 'wb') as object_result:
    pickle.dump(testresults, object_result, pickle.HIGHEST_PROTOCOL)

print('\n\n[OK]: Finished Script')

exit()


### 
## TEST LOSS AND ACCY EVOLUTION

### RESNET 56
#E = 3
#lab_ind = 'ResNet20'
#label_single = 'ResNet56'
#path_ = '../results/dicts/resnets/definitives/ResNet56/Results_Single_Models.pkl'
#path = '../results/dicts/resnets/definitives/ResNet56/Results_Ensemble_Models.pkl'


### RESNET 110
#E = 6
#lab_ind = 'ResNet20'
#label_single = 'ResNet110'
#path_ = '../results/dicts/resnets/definitives/ResNet110/Results_Single_Models.pkl'
#path = '../results/dicts/resnets/definitives/ResNet110/Results_Ensemble_Models.pkl'


### VGG 13
#E = 3
#lab_ind = 'VGG 9'
#label_single = 'VGG 13'
#path_ = '../results/dicts/vggs/definitives/VGG13/Results_Single_Models.pkl'
#path = '../results/dicts/vggs/definitives/VGG13/Results_Ensemble_Models.pkl'


### VGG 19
#E = 7
#lab_ind = 'VGG 9'
#label_single = 'VGG 19'
#path_ = '../results/dicts/vggs/definitives/VGG19/Results_Single_Models.pkl'
#path = '../results/dicts/vggs/definitives/VGG19/Results_Ensemble_Models.pkl'


## DENSENET 121
E = 6
lab_ind = 'Little DenseNet'
label_single = 'DenseNet 121'
path_ = '../results/dicts/densenets/definitives/densenet121/Results_Single.pkl'
path = '../results/dicts/densenets/definitives/densenet121/Results_Ensemble.pkl'


### DENSENET 169
#E = 9
#lab_ind = 'Little DenseNet'
#label_single = 'DenseNet 169'
#path_ = '../results/dicts/densenets/definitives/densenet169/Results_Single_Models.pkl'
#path = '../results/dicts/densenets/definitives/densenet169/Results_Ensemble_Models.pkl'




import pickle
with open(path_, 'rb') as input: results_ = pickle.load(input)
with open(path, 'rb') as input: results = pickle.load(input)

import matplotlib.pyplot as plt

psm = True
num_epochs = 350


c = [0, 'pink', 'blue', 'green', 'yellow', 'purple', 'brown', 'orange']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for m in range(1,1+E):
    ax1.plot(range(num_epochs), results.train_loss['m{}'.format(m)], label='{}_{}'.format(lab_ind, m), color=c[m], alpha=0.4)
ax1.plot(range(num_epochs), results.train_loss['ensemble'], label='Ensemble', color='black', alpha=1)
if psm: ax1.plot(range(num_epochs), results_.train_loss, label=label_single, color='red', alpha=1, linewidth=0.5)
ax1.set_title('Trianing Loss')
ax1.grid(True)

for m in range(1,1+E):
    ax2.plot(range(num_epochs), results.valid_loss['m{}'.format(m)], label='{}_{}'.format(lab_ind, m), color=c[m], alpha=0.4)
ax2.plot(range(num_epochs), results.valid_loss['ensemble'], label='Ensemble', color='black', alpha=1)
if psm: ax2.plot(range(num_epochs), results_.valid_loss, label=label_single, color='red', alpha=1, linewidth=0.5)
ax2.set_title('Validation Loss')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.grid(True)

for m in range(1,1+E):
    ax3.plot(range(num_epochs), results.train_accy['m{}'.format(m)], label='{}_{}'.format(lab_ind, m), color=c[m], alpha=0.4)
ax3.plot(range(num_epochs), results.train_accy['ensemble'], label='Ensemble', color='black', alpha=1)
if psm: ax3.plot(range(num_epochs), results_.train_accy, label=label_single, color='red', alpha=1, linewidth=0.5)
ax3.set_title('Training Accuracy')
ax3.grid(True)

for m in range(1,1+E):
    ax4.plot(range(num_epochs), results.valid_accy['m{}'.format(m)], label='{}_{}'.format(lab_ind, m), color=c[m], alpha=0.4)
ax4.plot(range(num_epochs), results.valid_accy['ensemble'], label='Ensemble', color='black', alpha=1)
if psm: ax4.plot(range(num_epochs), results_.valid_accy, label=label_single, color='red', alpha=1, linewidth=0.5)
ax4.set_title('Validation Accuracy')
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax4.grid(True)
plt.show()














