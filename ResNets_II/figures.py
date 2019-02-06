
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:56:06 2019
@author: pabloruizruiz
"""

import os
import pickle
import pandas as pd


## Introduce the correct path ##
path = 'ResNet56'
results_path = os.path.abspath('../results_II/dicts/resnets/' + path)  
assert os.path.exists(results_path), 'Results folder not found!'

# Data Wrapper
path_to_single = os.path.join(results_path, 'Results_Single_Models.pkl')
path_to_ensemble = os.path.join(results_path, 'Results_Ensemble_Models.pkl')
path_to_testing = os.path.join(results_path, 'Results_Testing.pkl')

try:
    with open(path_to_single, 'rb') as input:
        res = pickle.load(input)
except:
    print('No results for Single Model in {}', path_to_single)

try:
    with open(path_to_ensemble, 'rb') as input:
        eres = pickle.load(input)
except:
    print('No results for Ensemble Model in {}', path_to_ensemble)

try:
    with open(path_to_testing, 'rb') as input:
        test = pickle.load(input)
except:
    print('No results for Testing in {}', path_to_testing)
