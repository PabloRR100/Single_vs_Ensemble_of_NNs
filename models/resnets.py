
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    
    def __init__(self, name, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        
        self.name = name
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20():
    return ResNet('ResNet20', BasicBlock, [3, 3, 3])


def ResNet32():
    return ResNet('ResNet32', BasicBlock, [5, 5, 5])


def ResNet44():
    return ResNet('ResNet44', BasicBlock, [7, 7, 7])


def ResNet56():
    return ResNet('ResNet56', BasicBlock, [9, 9, 9])


def ResNet110():
    return ResNet('ResNet110', BasicBlock, [18, 18, 18])


def ResNet1202():
    return ResNet(BasicBlock, [200, 200, 200])


if __name__ == '__main__':

    import sys
    sys.path.append('..')
    
    import torch
    from utils import count_parameters
    from torch.autograd import Variable
    from beautifultable import BeautifulTable as BT

    resnet20 = ResNet20()
    resnet32 = ResNet32()
    resnet44 = ResNet44()
    resnet56 = ResNet56()
    resnet110 = ResNet110()
    
    table = BT()
    table.append_row(['Model', 'M. Paramars'])
    table.append_row(['ResNset20', count_parameters(resnet20)/1e6,])
    table.append_row(['ResNset32', count_parameters(resnet32)/1e6])
    table.append_row(['ResNset44', count_parameters(resnet44)/1e6])
    table.append_row(['ResNset56', count_parameters(resnet56)/1e6])
    table.append_row(['ResNset110', count_parameters(resnet110)/1e6])
    print(table)
        
    
    def test():
        net = ResNet56()
        y = net(Variable(torch.randn(1,3,32,32)))
        print(y.size())
    
    test()
    
    '''
    ResNets implemented on the paper <https://arxiv.org/pdf/1512.03385.pdf>
    
    +------------+-------------+-----------------+
    |   Model    | M. Paramars | % over ResNet20 |
    +------------+-------------+-----------------+
    | ResNet 20  |    0.27     |       1.0       |
    +------------+-------------+-----------------+
    | ResNet 32  |    0.464    |      1.721      |
    +------------+-------------+-----------------+
    | ResNet 44  |    0.659    |      2.442      |
    +------------+-------------+-----------------+
    | ResNet 56  |    0.853    |      3.163      |
    +------------+-------------+-----------------+
    | ResNet 110 |    1.728    |      6.406      |
    +------------+-------------+-----------------+
        
    '''
