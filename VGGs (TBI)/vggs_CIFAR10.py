
'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == '__main__':

    import sys
    sys.path.append('..')
    from utils import count_parameters
    from beautifultable import BeautifulTable as BT

    vgg11 = VGG('VGG11')
    vgg13 = VGG('VGG13')
    vgg16 = VGG('VGG16')
    vgg19 = VGG('VGG19')
    
    table = BT()
    table.append_row(['Model', 'M. of Params'])
    table.append_row(['VGG11', count_parameters(vgg11)*1e-6])
    table.append_row(['VGG13', count_parameters(vgg13)*1e-6])
    table.append_row(['VGG16', count_parameters(vgg16)*1e-6])
    table.append_row(['VGG19', count_parameters(vgg19)*1e-6])
    print(table)
        
    
    '''
    Source <https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py>
    
    +-------+--------------+
    | Model | M. of Params |
    +-------+--------------+
    | VGG11 |    9.231     |
    +-------+--------------+
    | VGG13 |    9.416     |
    +-------+--------------+
    | VGG16 |    14.728    |
    +-------+--------------+
    | VGG19 |    20.041    |
    +-------+--------------+
    
    '''
