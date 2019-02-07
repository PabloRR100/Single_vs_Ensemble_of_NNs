
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:56:06 2019
@author: pabloruizruiz
"""

import os
import pickle

## Introduce the correct path ##
path = 'ResNet56'
results_path = os.path.abspath('../results_II/dicts/resnets/' + path)  
assert os.path.exists(results_path), 'Results folder not found!'

# Data Wrapper
path_to_single = os.path.join(results_path, 'Results_Single_Models.pkl')
path_to_ensemble = os.path.join(results_path, 'Results_Ensemble_Models.pkl')
path_to_testing = os.path.join(results_path, 'Results_Testing.pkl')

if os.path.exists(path_to_single):
    with open(path_to_single, 'rb') as input:
        res = pickle.load(input)
else:
    print('No results for Single Model in {}', path_to_single)

if os.path.exists(path_to_ensemble):
    with open(path_to_ensemble, 'rb') as input:
        eres = pickle.load(input)
else:
    print('No results for Ensemble Model in {}', path_to_ensemble)

if os.path.exists(path_to_testing):
    with open(path_to_testing, 'rb') as input:
        test = pickle.load(input)
else:
    print('No results for Testing in {}', path_to_testing)




data1 = {'single':res.iter_train_accy, 
        'ensemble': eres.iter_train_accy['ensemble']}

data2 = {'single':res.iter_train_loss, 
        'ensemble': eres.iter_train_loss['ensemble']}

data3 = {'single':res.train_accy, 
        'ensemble': eres.train_accy['ensemble']}

data4 = {'single':res.train_loss, 
        'ensemble': eres.train_loss['ensemble']}

data5 = {'single':res.valid_accy, 
        'ensemble': eres.valid_accy['ensemble']}

data6 = {'single':res.valid_loss, 
        'ensemble': eres.valid_loss['ensemble']}


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def savefig(data: dict, path: str, title: str):
    ''' Save the plot from the data '''
    plt.figure()
    sns.set_style('darkgrid')
    df = pd.DataFrame.from_dict(data)
    sns.lineplot(data=df)
    plt.savefig(os.path.join(path, title))
 
title = 'ResNet56'
path_to_figures = '.'

sns.lineplot(data=pd.DataFrame.from_dict(res.valid_loss))
sns.lineplot(data=pd.DataFrame.from_dict(res.valid_accy))

savefig(data1, path_to_figures, title + '_train_accuracy_per_iter.png')
savefig(data2, path_to_figures, title + '_train_loss_per_iter.png')
savefig(data3, path_to_figures, title + '_train_accuracy_per_epoch.png')
savefig(data4, path_to_figures, title + '_train_loss_per_iter.png')
savefig(data5, path_to_figures, title + '_valid_accuracy.png')
savefig(data6, path_to_figures, title + '_valid_loss.png')