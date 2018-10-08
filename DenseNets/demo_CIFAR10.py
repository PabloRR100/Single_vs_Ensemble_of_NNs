
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'ImportWarning')
warnings.filterwarnings('ignore', 'DeprecationWarning')

import os
import glob
import pickle
import multiprocessing
from beautifultable import BeautifulTable as BT


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


import sys
sys.path.append('..')
sys.path.append('DenseNets')
from utils import load_dataset, count_parameters


''' 
CONFIGURATION 
-------------

Catch from the parser all the parameters to define the training
'''
print('\n\nCONFIGURATION')
print('-------------')

########################################################
from parser import args
save = args.save
name = args.name
draws = args.draws
dataset = args.dataset
testing = args.testing
comments = args.comments
ensemble_type = args.ensembleSize
n_epochs = args.epochs
n_iters = args.iterations
batch_size = args.batch_size
learning_rate = args.learning_rate
save_frequency = args.save_frequency
load_trained_models = args.pretrained

table = BT()
table.append_row(['Save', str(args.save)])
table.append_row(['Name', str(args.name)])
table.append_row(['Draws', str(args.draws)])
table.append_row(['Testing', str(args.testing)])
table.append_row(['Comments', str(args.comments)])
table.append_row(['Ensemble size', str(args.ensembleSize)])
if not load_trained_models:
    table.append_row(['-------------', '-------------'])
    table.append_row(['Epochs', n_epochs])
    table.append_row(['Iterations', n_iters])
    table.append_row(['Batch Size', batch_size])
    table.append_row(['Learning Rate', str(args.learning_rate)])
else:
    table.append_row(['-------------', '-------------'])
    table.append_row(['No Training', 'Pretrained Models'])
print(table)
#########################################################



########################################################
## Backup code to debug from python shell - no parser
#save = False                # Activate results saving 
#draws = False               # Activate showing the figures
#dataset = 'CIFAR10'
#testing = True             # Activate test to run few iterations per epoch       
#comments = True             # Activate printing comments
#createlog = False           # Activate option to save the logs in .txt
#save_frequency = 1          # After how many epochs save stats
#ensemble_type = 'Big'       # Single model big 
##ensemble_type = 'Huge'     # Single model huge
#learning_rate = 0.1
#batch_size = 64
#n_iters = 106000
#load_trained_models = False # Load pretrained models instead of training
########################################################


momentum = 0.9
weight_decay = 1e-4

#n_epochs = int(n_iters / batch_size)

# GPU if CUDA is available
cuda = torch.cuda.is_available()
n_workers = multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = True if torch.cuda.device_count() > 1 else False
mem = False if device == 'cpu' else True


table = BT()
table.append_row(['Python Version', sys.version[:5]])
table.append_row(['PyTorch Version', torch.__version__])
table.append_row(['Cuda', str(cuda)])
table.append_row(['Device', str(device)])
table.append_row(['Cores', str(n_workers)])
table.append_row(['GPUs', str(torch.cuda.device_count())])
table.append_row(['CUDNN Enabled', str(torch.backends.cudnn.enabled)])
print('\n\nCOMPUTING CONFIG')
print('----------------')
print(table)



'''
DEFININTION OF PATHS 
--------------------
Define all the paths to load / save files
Ensure all those paths are correctly defined before moving on
'''

print('DEFINITION OF PATHS')
print('-------------------')
scripts = os.getcwd()
root = os.path.abspath(os.path.join(scripts, '../'))
results = os.path.abspath(os.path.join(root, 'results'))
data_path = os.path.abspath(os.path.join(root, '../datasets'))

path_to_logs = os.path.join(results, 'logs', 'densenets')
path_to_models = os.path.join(results, 'models', 'densenets')
path_to_figures = os.path.join(results, 'figures', 'densenets')

train_log = os.path.join(path_to_logs, 'train')
test_log = os.path.join(path_to_logs, 'test')

assert os.path.exists(root), 'Root folder not found'
assert os.path.exists(scripts), 'Scripts folder not found'
assert os.path.exists(results), 'Results folder not found'
assert os.path.exists(data_path), 'Data folder not found'
assert os.path.exists(path_to_logs), 'Logs folder not found'
assert os.path.exists(path_to_models), 'Models folder not found'
assert os.path.exists(path_to_figures), 'Figure folder not found'

print('Paths Validated')
print('---------------')
print('Root path: ', root)
print('Script path: ', scripts)
print('Result path: ', results)
print('DataFolder path: ', data_path)

paths = {
    'root': root, 
    'script': scripts,
    'data': data_path,
    'resulsts': results,
    'logs': {'train': train_log, 'test': test_log}, 
    'models': path_to_models,
    'figures': path_to_figures
}



# 1 - Import the Dataset
# ----------------------

print('IMPORTING DATA')
print('--------------')


train_set, valid_set, test_set = load_dataset(data_path, dataset, comments=comments)

train_loader = DataLoader(dataset = train_set.dataset, 
                          sampler=SubsetRandomSampler(train_set.indices),
                          batch_size = batch_size, num_workers=n_workers,
                          pin_memory = mem)

valid_loader = DataLoader(dataset = valid_set.dataset, 
                          sampler=SubsetRandomSampler(valid_set.indices),
                          batch_size = batch_size, num_workers=n_workers,
                          pin_memory = mem)

test_loader = DataLoader(dataset = test_set, batch_size = 1,
                         shuffle = False, num_workers=n_workers, pin_memory = mem)


batches = len(train_loader)
samples = len(train_loader.sampler.indices) 
n_iters = int(n_epochs * batches)



# 2 - Import the DenseNets
# ------------------------

print('\n\nIMPORTING MODELS')
print('----------------')

from densenets_Paper import (denseNet_40_12, denseNet_100_12, denseNet_100_24, 
                              denseNetBC_100_12, denseNetBC_250_24, denseNetBC_190_40)

densenet_40_12 = denseNet_40_12()
densenet_100_12 = denseNet_100_12()
densenet_100_24 = denseNet_100_24()
densenetBC_100_12 = denseNetBC_100_12() 
densenetBC_250_24 = denseNetBC_250_24()
densenetBC_190_40 = denseNetBC_190_40()


def parameters(model, typ=None):
    def compare_to_simplest(model, typ):
        simplest1 = count_parameters(densenet_40_12)
        simplest2 = count_parameters(densenetBC_100_12)
        if typ is None: return count_parameters(model) / simplest1
        if typ == 'BC': return count_parameters(model) / simplest2
    return count_parameters(model)*1e-6, compare_to_simplest(model, typ)


table = BT()
table.append_row(['Model', 'k', 'L', 'M. of Params', '% Over simplest'])
table.append_row(['DenseNet', 12, 40, *parameters(densenet_40_12)])
table.append_row(['DenseNet', 12, 100, *parameters(densenet_100_12)])
table.append_row(['DenseNet', 24, 100, *parameters(densenet_100_24)])
table.append_row(['DenseNet-BC', 12, 100, *parameters(densenetBC_100_12, 'BC')])
table.append_row(['DenseNet-BC', 24, 250, *parameters(densenetBC_250_24, 'BC')])
table.append_row(['DenseNet-BC', 40, 190, *parameters(densenetBC_190_40, 'BC')])
if comments: print(table)


# Apply constraint - Parameters constant

small = count_parameters(denseNetBC_100_12())  # 19:1 vs 33:1
singleModel = denseNetBC_250_24() if ensemble_type == 'Big' else denseNetBC_190_40() 
ensemble_size = round(count_parameters(singleModel) / small)

# Construct the single model

singleModel = denseNetBC_250_24() if ensemble_type == 'Big' else denseNetBC_190_40() # 3:1 vs 6:1
title = singleModel.name

name = singleModel.name
singleModel.to(device)
if gpus: singleModel = nn.DataParallel(singleModel)
optimizer = optim.SGD(singleModel.parameters(), learning_rate, momentum, weight_decay)


# Construct the ensemble

names = []
ensemble = []
optimizers = []
for i in range(ensemble_size):
    
    model = densenetBC_100_12()
    names.append(model.name + '_' + str(i+1))
    params = optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    optimizers.append(params)
    
    model.to(device)
    if gpus: model = nn.DataParallel(model)
    ensemble.append(model)

# Construct the ensemble

ensemble = []

for i in range(ensemble_size):
    model = denseNetBC_100_12()
    model.name = model.name + '_' + str(i+1)
    ensemble.append(model)



# 3 - Train DenseNet
# ------------------
    
    if load_trained_models:
    
    ## LOAD TRAINED MODELS
    print('Loading trained models')
    
    def loadmodel(model, device, path):
        return model.load_state_dict(torch.load(path, map_location=device))
                    
    # Load saved models
    ps = glob.glob(os.path.join(paths['models'], '*.pkl'))
    
    # Single Model
    singleModel = loadmodel(singleModel, device, ps[0])
    
    # Ensemble Members
    ensemble = []
    for p in ps[1:]:
        model = loadmodel(singleModel, device, p)
        ensemble.append(model)
            
else:
    
    ## TRAINING   
    
    print('\n\nTRAINING')
    print('--------')
    
    criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
    
    # Big Single Model
    
    cudnn.benchmark = False    
    cudnn.benchmark = True
    from train import train
    print('Starting Single Model Training...' )
    
    params = [dataset, name, singleModel, optimizer, criterion, device, train_loader,
              valid_loader, n_epochs, n_iters, save, paths, testing]
    
    results = train(*params)
    with open('Results_Single_Models.pkl', 'wb') as object_result:
        pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL)
    
    results.show()
    
    
    # Ensemble Model
    
    cudnn.benchmark = False    
    cudnn.benchmark = True
    from train_ensemble import train as train_ensemble
    print('Starting Ensemble Training...')
    
    params = [dataset, names, ensemble, optimizers, criterion, device, train_loader,
              valid_loader, n_epochs, n_iters, save, paths, testing]
        
    ens_results = train_ensemble(*params)
    with open('Results_Ensemble_Models.pkl', 'wb') as object_result:
        pickle.dump(ens_results, object_result, pickle.HIGHEST_PROTOCOL)
    
    ens_results.show()


# 4 - Evaluate Models
# -------------------
    
print('\n\nTESTING')
print('-------')

from test import test
    
testresults = test('CIFAR10', name, singleModel, ensemble, device, test_loader, paths, save)
with open('Results_Testing.pkl', 'wb') as object_result:
    pickle.dump(testresults, object_result, pickle.HIGHEST_PROTOCOL)



exit()
