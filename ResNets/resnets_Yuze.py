
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MotherBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, planes_factor=1, size_incre=0, need_sc=False):
        super(MotherBlock, self).__init__()
        self.head = nn.Sequential()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or need_sc:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.head(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, planes_factor=1, size_incre=0, need_sc=False):
        super(BasicBlock, self).__init__()
        first_layer_planes = planes * (planes_factor if stride == 1 else 1)
        first_layer_size = 3 + (size_incre if stride == 1 else 0)
        first_layer_padding = 1 + (size_incre / 2 if stride == 1 else 0)
        self.conv1 = nn.Conv2d(in_planes, first_layer_planes, kernel_size=first_layer_size, stride=stride, padding=first_layer_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(first_layer_planes)
        self.conv2 = nn.Conv2d(first_layer_planes, planes * planes_factor, kernel_size=3+size_incre, stride=1, padding=1+size_incre/2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * planes_factor)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes * planes_factor or need_sc:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes * planes_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes * planes_factor)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def to_zero(self):
        self.conv1.weight.data.fill_(0)
        self.conv2.weight.data.fill_(0)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)
        self.bn1.running_mean.fill_(0)
        self.bn1.running_var.fill_(1)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)
        self.bn2.running_mean.fill_(0)
        self.bn2.running_var.fill_(1)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, planes_factor=1, size_incre=0, need_sc=False):
        super(Bottleneck, self).__init__()
        second_layer_planes = planes * (planes_factor if stride == 1 else 1)
        second_layer_size = 3 + (size_incre if stride == 1 else 0)
        second_layer_padding = 1 + (size_incre / 2 if stride == 1 else 0)
        self.conv1 = nn.Conv2d(in_planes, planes * planes_factor, kernel_size=1+size_incre, padding=size_incre/2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * planes_factor)
        self.conv2 = nn.Conv2d(planes * planes_factor, second_layer_planes, kernel_size=second_layer_size, stride=stride, padding=second_layer_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(second_layer_planes)
        self.conv3 = nn.Conv2d(second_layer_planes, self.expansion * planes * planes_factor, kernel_size=1 + size_incre, padding=size_incre/2, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes * planes_factor)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes * planes_factor or need_sc:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes * planes_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes * planes_factor)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def to_zero(self):
        self.conv1.weight.data.fill_(0)
        self.conv2.weight.data.fill_(0)
        self.conv3.weight.data.fill_(0)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)
        self.bn1.running_mean.fill_(0)
        self.bn1.running_var.fill_(1)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)
        self.bn2.running_mean.fill_(0)
        self.bn2.running_var.fill_(1)
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.fill_(0)
        self.bn3.running_mean.fill_(0)
        self.bn3.running_var.fill_(1)


class ResNet(nn.Module):
    # num_blocks is indeed the setting of blocks
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512 * block.expansion * num_blocks[3][1], 1000)
        self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks[0] - 1)
        layers = []
        first_try = 0
        for stride in strides:
            if first_try == 0:
                first_try += 1
                if planes == 64:
                    layers.append(block(self.in_planes, planes, stride, num_blocks[1], num_blocks[2], need_sc=True))
                    self.in_planes = planes * block.expansion * num_blocks[1]
                    continue
            layers.append(block(self.in_planes, planes, stride, num_blocks[1], num_blocks[2]))
            self.in_planes = planes * block.expansion * num_blocks[1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


def ResNet18(Block_Setting):
    return ResNet(BasicBlock, Block_Setting)


def ResNet34(Block_Setting):
    return ResNet(BasicBlock, Block_Setting)


def ResNet50(Block_Setting):
    return ResNet(Bottleneck, Block_Setting)


def ResNet101(Block_Setting):
    return ResNet(Bottleneck, Block_Setting)


def ResNet152(Block_Setting):
    return ResNet(Bottleneck, Block_Setting)

def MotherNet():
    return ResNet(MotherBlock, [[2, 1, 0]] * 4)



if __name__ == '__main__':

    import copy
    models = []
    model_name = ['MN', 'Res18_0', 'Res18_1', 'Res18_2','Res18_3', 'Res18_4',       \
                        'Res34_0', 'Res34_1', 'Res34_2','Res34_3', 'Res34_4']

    base_block_settings = [[[2, 1, 0]] * 4,                                 \
                           [[3, 1, 0], [4, 1, 0], [6, 1, 0], [3, 1, 0]],    \
                           [[3, 1, 0], [4, 1, 0], [6, 1, 0], [3, 1, 0]],    \
                           [[3, 1, 0], [4, 1, 0], [23, 1, 0], [3, 1, 0]],   \
                           [[3, 1, 0], [8, 1, 0], [36, 1, 0], [3, 1, 0]]]

    models.append(ResNet18([[2, 1, 0]] * 4))

    for i in range(0, 2):
        for j in range(0, 5):
            tmp_settings = copy.deepcopy(base_block_settings[i])
            if j == 1:
                tmp_settings[0][1] = 2
                tmp_settings[2][1] = 2
            elif j == 2:
                tmp_settings[1][1] = 2
                tmp_settings[3][1] = 2
            elif j == 3:
                tmp_settings[0][2] = 2
                tmp_settings[2][2] = 2
            elif j == 4:
                tmp_settings[1][2] = 2
                tmp_settings[3][2] = 2

            if i == 0:
                models.append(ResNet18(tmp_settings))
            elif i == 1:
                models.append(ResNet34(tmp_settings))
            elif i == 2:
                models.append(ResNet50(tmp_settings))
            elif i == 3:
                models.append(ResNet101(tmp_settings))
            elif i == 4:
                models.append(ResNet152(tmp_settings))
    
    
    
    import sys
    sys.path.append('..')
    from utils import count_parameters
    from beautifultable import BeautifulTable as BT

    resnet18 = ResNet18([[2, 1, 0]] * 4)
    resnet34 = ResNet34()
    resnet50 = ResNet50()
    resnet101 = ResNet101()
    resnet152 = ResNet152()
    
    table = BT()
    table.append_row(['Model', 'M. Paramars'])
    table.append_row(['ResNset20', count_parameters(resnet18)/1e6,])
    table.append_row(['ResNset32', count_parameters(resnet34)/1e6])
    table.append_row(['ResNset44', count_parameters(resnet50)/1e6])
    table.append_row(['ResNset56', count_parameters(resnet101)/1e6])
    table.append_row(['ResNset110', count_parameters(resnet152)/1e6])
    print(table)
        
    
    def test():
        net = ResNet50()
        y = net(Variable(torch.randn(1,3,32,32)))
        print(y.size())
    
    test()
    
    '''
    ResNetss implemented on the paper <https://arxiv.org/pdf/1512.03385.pdf>
    
    +------------+-------------+
    |   Model    | M. Paramars |
    +------------+-------------+
    | ResNset20  |    0.272    |
    +------------+-------------+
    | ResNset32  |    0.467    |
    +------------+-------------+
    | ResNset44  |    0.661    |
    +------------+-------------+
    | ResNset56  |    0.856    |
    +------------+-------------+
    | ResNset110 |    1.731    |
    +------------+-------------+
    
    '''


resnet18 = ResNet18([[2, 1, 0]] * 4)
count_parameters(resnet18) * 1e-6


