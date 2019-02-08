#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tom Goldstein
@github: https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/models/vgg.py
"""

import torch
import torch.nn as nn

cfg = {
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    '''
    VGG with Batch Nomalization
    '''
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.name = vgg_name
        self.input_size = 32
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier = nn.Linear(self.n_maps, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
                   nn.BatchNorm1d(self.n_maps),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG9():
    return VGG('VGG9')

def VGG11():
    return VGG('VGG11')

def VGG13():
    return VGG('VGG13')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')

if __name__ == '__main__':
    
    import sys
    sys.path.append('..')
    from utils import count_parameters
    from torch.autograd import Variable    
    from beautifultable import BeautifulTable as BT

    vgg9  = VGG('VGG9')
    vgg11 = VGG('VGG11')
    vgg13 = VGG('VGG13')
    vgg16 = VGG('VGG16')
    vgg19 = VGG('VGG19')
    
    table = BT()
    table.append_row(['Model', 'M. of Params'])
    table.append_row(['VGG9', count_parameters(vgg11)*1e-6])
    table.append_row(['VGG11', count_parameters(vgg11)*1e-6])
    table.append_row(['VGG13', count_parameters(vgg13)*1e-6])
    table.append_row(['VGG16', count_parameters(vgg16)*1e-6])
    table.append_row(['VGG19', count_parameters(vgg19)*1e-6])
    print(table)
    
    from utils import timeit
    
    @timeit
    def test(net):
        y = net(Variable(torch.randn(128,3,32,32)))
        print(y.size())
    
    test(vgg9)  # ''' 1440.98 ms '''
    test(vgg19) # ''' 2344.25 ms '''
        
    
    '''
    +-------+--------------+
    | Model | M. of Params |
    +-------+--------------+
    | VGG9  |     2.79     |
    +-------+--------------+
    | VGG11 |    9.495     |
    +-------+--------------+
    | VGG13 |     9.68     |
    +-------+--------------+
    | VGG16 |    14.992    |
    +-------+--------------+
    | VGG19 |    20.304    |
    +-------+--------------+
    '''