
#!/usr/bin/python2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from scipy.optimize import minimize
import ConfigParser
import sys
import time
import copy
import os
import argparse
import tqdm
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict
# from guppy import hpy
from memory_profiler import profile
import gc


class NN(nn.Module):
    def __init__(self, NN_part1, NN_part2, NN_view_size):
        super(NN, self).__init__()
        self.part1 = nn.Sequential(*NN_part1)
        self.part2 = nn.Sequential(*NN_part2)
        self.view_size = NN_view_size
        self._initialize_weights()

    def forward(self, x):
        x = self.part1(x)
        x = x.view(-1, self.view_size)
        x = self.part2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Reference: https://github.com/erogol/Net2Net/blob/master/net2net.py
    def Net2WiderNet(self, current_layer, next_layer, new_width, bnorm=None, noise=True):
        w1 = current_layer.weight.data
        w2 = next_layer.weight.data
        b1 = current_layer.bias.data

        if 'Conv' in current_layer.__class__.__name__ or 'Linear' in next_layer.__class__.__name__:
            if 'Conv' in current_layer.__class__.__name__ and 'Linear' in next_layer.__class__.__name__:
                channel_length = int(np.sqrt(w2.size(1) / w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1) / channel_length ** 2, channel_length, channel_length)
            old_width = w1.size(0)

            if 'Conv' in current_layer.__class__.__name__:
                new_w1 = torch.FloatTensor(new_width, w1.size(1), w1.size(2), w1.size(3))
                new_w2 = torch.FloatTensor(new_width, w2.size(0), w2.size(2), w2.size(3))
            else:
                new_w1 = torch.FloatTensor(new_width, w1.size(1))
                new_w2 = torch.FloatTensor(new_width, w2.size(0))
            new_b1 = torch.FloatTensor(new_width)

            if bnorm is not None:
                new_norm_mean = torch.FloatTensor(new_width)
                new_norm_var = torch.FloatTensor(new_width)
                if bnorm.affine:
                    new_norm_weight = torch.FloatTensor(new_width)
                    new_norm_bias = torch.FloatTensor(new_width)

            w2.transpose_(0, 1)
            new_w1.narrow(0, 0, old_width).copy_(w1)
            new_w2.narrow(0, 0, old_width).copy_(w2)
            new_b1.narrow(0, 0, old_width).copy_(b1)

            if bnorm is not None:
                new_norm_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
                new_norm_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
                if bnorm.affine:
                    new_norm_weight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                    new_norm_bias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

            index_set = dict()
            for i in range(old_width, new_width):
                sampled_index = np.random.randint(0, old_width)
                if sampled_index in index_set:
                    index_set[sampled_index].append(i)
                else:
                    index_set[sampled_index] = [sampled_index]
                    index_set[sampled_index].append(i)
                new_w1.select(0, i).copy_(w1.select(0, sampled_index).clone())
                new_w2.select(0, i).copy_(w2.select(0, sampled_index).clone())
                new_b1[i] = b1[sampled_index]
                if bnorm is not None:
                    new_norm_mean[i] = bnorm.running_mean[sampled_index]
                    new_norm_var[i] = bnorm.running_var[sampled_index]
                    if bnorm.affine:
                        new_norm_weight[i] = bnorm.weight.data[sampled_index]
                        new_norm_bias[i] = bnorm.bias.data[sampled_index]
            for (index, d) in index_set.items():
                div_length = len(d)
                for next_layer_index in d:
                    new_w2[next_layer_index].div_(div_length)
            current_layer.out_channels = new_width
            next_layer.in_channels = new_width

            if noise:
                w1_added_noise = np.random.normal(scale=5e-2 * new_w1.std(), size=list(new_w1.size()))
                new_w1 += torch.FloatTensor(w1_added_noise).type_as(new_w1)
                w2_added_noise = np.random.normal(scale=5e-2 * new_w2.std(), size=list(new_w2.size()))
                new_w2 += torch.FloatTensor(w2_added_noise).type_as(new_w2)
            new_w1.narrow(0, 0, old_width).copy_(w1)
            new_w2.narrow(0, 0, old_width).copy_(w2)
            for (index, d) in index_set.items():
                div_length = len(d)
                new_w2[index].div_(div_length)

            w2.transpose_(0, 1)
            new_w2.transpose_(0, 1)
            current_layer.weight.data = new_w1
            current_layer.bias.data = new_b1

            if 'Conv' in current_layer.__class__.__name__ and 'Linear' in next_layer.__class__.__name__:
                next_layer.weight.data = new_w2.view(next_layer.weight.data.size(0), new_width * channel_length ** 2)
                next_layer.in_features = new_width * channel_length ** 2
            else:
                next_layer.weight.data = new_w2

            if bnorm is not None:
                bnorm.running_var = new_norm_var
                bnorm.running_mean = new_norm_mean
                if bnorm.affine:
                    bnorm.weight.data = new_norm_weight
                    bnorm.bias.data = new_norm_bias

    def Net2DeeperNet(self, current_layer_id, new_layer_fil_size=1, new_layer_type='Conv', noise=True):
        current_layer = self.part1[current_layer_id]
        if "Linear" in new_layer_type:
            new_layer = nn.Linear(current_layer.out_features, current_layer.out_features)
            new_layer.weight.data.copy_(torch.eye(current_layer.out_features))
            new_layer.bias.data.zero_()
            bnorm = nn.BatchNorm1d(current_layer.out_features)
            bnorm.weight.data.fill_(1)
            bnorm.bias.data.fill_(0)
            bnorm.running_mean.fill_(0)
            bnorm.running_var.fill_(1)
        elif "Conv" in new_layer_type:
            new_kernel_size = new_layer_fil_size
            new_layer = nn.Conv2d(current_layer.out_channels,        \
                                  current_layer.out_channels,        \
                                  kernel_size=new_kernel_size,       \
                                  padding=(new_kernel_size - 1) / 2)
            new_layer.weight.data.zero_()
            center = new_layer.kernel_size[0] // 2 + 1
            for i in range(0, current_layer.out_channels):
                new_layer.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, center - 1, 1).narrow(3, center - 1, 1).fill_(1)
            if noise:
                added_noise = np.random.normal(scale=5e-2 * new_layer.weight.data.std(), size=list(new_layer.weight.size()))
                new_layer.weight.data += torch.FloatTensor(added_noise).type_as(new_layer.weight.data)
            new_layer.bias.data.zero_()
            bnorm = nn.BatchNorm2d(new_layer.out_channels)
            bnorm.weight.data.fill_(1)
            bnorm.bias.data.fill_(0)
            bnorm.running_mean.fill_(0)
            bnorm.running_var.fill_(1)

        sub_part1 = list(self.part1.children())[0: current_layer_id + 3]
        sub_part2 = list(self.part1.children())[current_layer_id + 3:]
        sub_part1.append(new_layer)
        sub_part1.append(bnorm)
        sub_part1.append(nn.ReLU())
        sub_part1.extend(sub_part2)
        self.part1 = nn.Sequential(*sub_part1)

    def Net2LongerNet(self, current_layer, new_length, noise=True):
        w1 = current_layer.weight.data
        half_length_increment = (new_length - w1.size(2)) / 2
        new_w1 = torch.FloatTensor(w1.size(0), w1.size(1), new_length, new_length).zero_()
        new_w1.narrow(2, half_length_increment, w1.size(2)).narrow(3, half_length_increment, w1.size(3)).copy_(w1)
        if noise:
            added_noise = np.random.normal(scale=5e-2 * new_w1.std(), size=list(new_w1.size()))
            new_w1 += torch.FloatTensor(added_noise).type_as(new_w1)
        new_w1.narrow(2, half_length_increment, w1.size(2)).narrow(3, half_length_increment, w1.size(3)).copy_(w1)
        current_layer.weight.data = new_w1
        current_layer.kernel_size = (new_length, new_length)
        current_layer.padding = ((new_length - 1) / 2, (new_length - 1) / 2)


def function_preserving_transfer(mother, child):
    '''
    Transfer the MotherNet model to the same network size as the child model using the function-preserving transformation method
    :param mother: original MotherNet model
    :param child: child model to be trained
    :return: transfered MotherNet model
    '''
    network_len = len(child.part1)
    for i in range(0, network_len):
        mother_layer_type = mother.part1[i].__class__.__name__
        child_layer_type = child.part1[i].__class__.__name__

        if 'Conv' not in child_layer_type:
            continue
        else:
            if 'Conv' in mother_layer_type:
                mother_kernel_size = mother.part1[i].kernel_size
                mother_out_channels = mother.part1[i].out_channels
                child_kernel_size = child.part1[i].kernel_size
                child_out_channels = child.part1[i].out_channels
                if mother_kernel_size != child_kernel_size:
                    mother.Net2LongerNet(mother.part1[i], child_kernel_size[0])
                if mother_out_channels != child_out_channels:
                    mother_network_len = len(mother.part1)
                    for pos in range(i + 1, mother_network_len):
                        if 'Conv' in mother.part1[pos].__class__.__name__:# or 'Linear' in mother.part1[pos].__class__.__name__:
                            mother.Net2WiderNet(mother.part1[i], mother.part1[pos], child_out_channels, bnorm=mother.part1[i+1])
                            break
                        elif pos == mother_network_len - 1:
                            mother.Net2WiderNet(mother.part1[i], mother.part2[0], child_out_channels, bnorm=mother.part1[i+1])
                            mother.view_size *= child_out_channels / mother_out_channels
            else:
                child_kernel_size = child.part1[i].kernel_size
                child_out_channels = child.part1[i].out_channels
                mother.Net2DeeperNet(i - 3)
                mother_kernel_size = mother.part1[i].kernel_size
                mother_out_channels = mother.part1[i].out_channels
                if mother_kernel_size != child_kernel_size:
                    mother.Net2LongerNet(mother.part1[i], child_kernel_size[0])
                if mother_out_channels != child_out_channels:
                    mother_network_len = len(mother.part1)
                    for pos in range(i + 1, mother_network_len):
                        if 'Conv' in mother.part1[pos].__class__.__name__:
                            mother.Net2WiderNet(mother.part1[i], mother.part1[pos], child_out_channels, bnorm=mother.part1[i+1])
                            break
                        elif pos == mother_network_len - 1:
                            mother.Net2WiderNet(mother.part1[i], mother.part2[0], child_out_channels, bnorm=mother.part1[i+1])
                            mother.view_size *= child_out_channels / mother_out_channels
    return mother


# Reference: https://github.com/jucheng1992/pySL/blob/master/pySL.py
def nll_SL(w, new_training, new_label, model_num):
    for i in range(model_num):
        if i == 0:
            f_new = w[i] * new_training[i, :, :]
        else:
            f_new = f_new + w[i] * new_training[i, :, :]
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss_value = loss(Variable(torch.from_numpy(f_new)), Variable(torch.from_numpy(new_label))).item()
    return loss_value


def super_learner(models):
    cuda_enabled = torch.cuda.is_available()
    end_index = int(6000) / 500 # 6000
    for idx, (data, target) in enumerate(SL_loader):
        if idx == end_index:
            break
        if idx == 0:
            new_label = target.numpy()
        else:
            new_label = np.concatenate([new_label, target.numpy()])
        with torch.no_grad():
            data = Variable(data)
        if cuda_enabled:
            data = data.cuda()
        for i in range(len(models)):
            models[i].eval()
            if cuda_enabled:
                models[i].cuda()
            output = models[i](data).data.cpu().numpy().reshape(-1, len(target), 10)
            if i == 0:
                inner_new_training = output
            else:
                inner_new_training = np.row_stack((inner_new_training, output))
            models[i].cpu()
        if idx == 0:
            new_training = inner_new_training
        else:
            new_training = np.column_stack([new_training, inner_new_training])
    w0 = np.array([1.0 / len(models)] * len(models))
    return minimize(nll_SL, w0, args=(new_training, new_label, len(models)), method='SLSQP')


def train(model, train_loader, learning_rate):
    cuda_enabled = torch.cuda.is_available()
    model.train()
    if cuda_enabled:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    total_acc = 0
    begin_t = int(round(time.time() * 1000))
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), ascii=True, total=len(train_loader)):
        if cuda_enabled:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()
        output_ = loss(output, target)
        pred = output.data.max(1)[1]
        correct_prediction = pred.eq(target.data).cpu().sum().data.numpy()
        output_.backward()
        optimizer.step()

        training_acc = 1.0 * correct_prediction / len(target)
        batch_num = batch_idx + 1
        total_acc += training_acc

        if batch_idx % 24 == 0 and batch_idx != 0:
            form_train_writer.writerow(['%g'% (total_acc / batch_num)])

    end_t = int(round(time.time() * 1000))
    print('epoch time: %ds' % ((end_t - begin_t)/1000.0))
    print('epoch acc: %g'% (total_acc / batch_num))
    return ((end_t - begin_t)/1000.0), (total_acc / batch_num)


def test(models, config_files=[], path=False, print_acc=True, super_learn=True):
    cuda_enabled = torch.cuda.is_available()
    if super_learn:
        meta_learner = super_learner(models)
        w = meta_learner['x']
    else:
        w = np.array([1.0 / len(models)] * len(models))

    SL_correct_prediction = 0
    averaging_correct_prediction = 0
    oracle_correct_prediction = 0
    voting_correct_prediction = 0
    current_model_correct_prediction = 0

    for data, target in test_loader:
        if cuda_enabled:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)

        for i in range(len(models)):
            if path:
                _, model = config_file2model(config_files[i], store_model=False)
                model.load_state_dict(state_dict_norm(torch.load(models[i])))
            else:
                model = models[i]
            model.eval()
            if cuda_enabled:
                model.cuda()

            output_before_softmax = model(data)
            output_after_softmax = F.softmax(output_before_softmax, dim=1)
            output_before_softmax = output_before_softmax.data.cpu().numpy().reshape(-1, len(target), 10)
            pred_single = output_after_softmax.data.max(1)[1]
            pred_tmp = pred_single.cpu().view(-1, 1)
            onehot_tmp = torch.FloatTensor(len(target), 10)
            onehot = onehot_tmp.zero_().scatter_(1, pred_tmp, 1)
            if i == 0:
                if super_learn:
                    SL_matrix = output_before_softmax
                voting_output = onehot.detach().numpy()
                averaging_output = output_after_softmax.detach().cpu().numpy()
            else:
                if super_learn:
                    SL_matrix = np.row_stack((SL_matrix, output_before_softmax))
                voting_output = voting_output + onehot.detach().numpy()
                averaging_output = averaging_output + output_after_softmax.detach().cpu().numpy()
            if i == len(models) - 1:
                current_model_output = output_after_softmax

            del model
            del output_before_softmax
            del output_after_softmax
            del pred_single
            del pred_tmp
            del onehot_tmp
            del onehot
            gc.collect()

        if super_learn:
            for i in range(len(models)):
                if i == 0:
                    SL_output = w[i] * SL_matrix[i, :, :]
                else:
                    SL_output = SL_output + w[i] * SL_matrix[i, :, :]

        oracle_output = torch.from_numpy(voting_output[range(len(target)), target.data.cpu().numpy()])
        if super_learn:
            with torch.no_grad():
                SL_output = Variable(torch.from_numpy(SL_output))
            if cuda_enabled:
                SL_output = SL_output.cuda()
            SL_pred = SL_output.data.max(1)[1]
        with torch.no_grad():
            voting_output = Variable(torch.from_numpy(voting_output))
        averaging_output = torch.from_numpy(averaging_output)
        if cuda_enabled:
            voting_output = voting_output.cuda()
            averaging_output = averaging_output.cuda()

        voting_pred = voting_output.data.max(1)[1]
        averaging_pred = averaging_output.data.max(1)[1]
        current_model_pred = current_model_output.data.max(1)[1]
        current_model_correct_prediction += current_model_pred.eq(target.data).cpu().sum().data.numpy()
        voting_correct_prediction += voting_pred.eq(target.data).cpu().sum().data.numpy()
        oracle_nonzero_vec = torch.nonzero(oracle_output).size()
        if oracle_nonzero_vec:
            oracle_correct_prediction += oracle_nonzero_vec[0]
        if super_learn:
            SL_correct_prediction += SL_pred.eq(target.data).cpu().sum().data.numpy()
        averaging_correct_prediction += averaging_pred.eq(target.data).cpu().sum().data.numpy()

        if super_learn:
            del SL_matrix
            del SL_output
            del SL_pred
        del voting_output
        del averaging_output
        del current_model_output
        del oracle_output
        del current_model_pred
        del voting_pred
        del averaging_pred
        del oracle_nonzero_vec
        gc.collect()

    if print_acc:
        print('averaging test accuracy: %g' % (1.0 * averaging_correct_prediction / len(test_loader.dataset)))
        print('oracle test accuracy: %g' % (1.0 * oracle_correct_prediction / len(test_loader.dataset)))
        print('voting test accuracy: %g' % (1.0 * voting_correct_prediction / len(test_loader.dataset)))
        if super_learn:
            print('SL test accuracy: %g' % (1.0 * SL_correct_prediction / len(test_loader.dataset)))
        print('current model test accuracy: %g' % (1.0 * current_model_correct_prediction / len(test_loader.dataset)))

    return (1.0 * averaging_correct_prediction / len(test_loader.dataset))


class Config_Net(object):
    batch_size = 128
    learning_rate = 0.001
    epoch_num = 10
    early_stop_threshold = 20


def config_file2model(config_file, store_model=True):
    '''
    Generate the network models based on the configuration files
    :param config_files: configuration files
    :return: generated models
    '''
    config_net = Config_Net()
    NN_part1 = []
    NN_part2 = []
    model_name = config_file.split('/')[-1][:-5]

    config = ConfigParser.ConfigParser()
    config.read(config_file)
    config_items = config.items('hyperparameter')

    last_layer = 'input'
    last_layer_output_num = 3
    pooling_scale = 1
    for item in config_items:
        if 'layer' in item[0]:
            layer_name = item[1].strip()
            if 'conv' in layer_name:
                dash = layer_name.index('-')
                filter_size = int(layer_name[dash + 1: len(layer_name)])
                kernel_size = int(layer_name[4: dash])
                tmp_conv2d = nn.Conv2d(last_layer_output_num, \
                                       filter_size, \
                                       kernel_size=kernel_size, \
                                       padding=(kernel_size - 1) / 2)
                tmp_conv2d.weight.data.normal_(0, 0.01)
                tmp_conv2d.bias.data.fill_(0)
                NN_part1.append(tmp_conv2d)
                NN_part1.append(nn.BatchNorm2d(filter_size))
                NN_part1.append(nn.ReLU())
                last_layer = 'conv'
                last_layer_output_num = filter_size
            elif 'pool' in layer_name:
                pooling_size = int(layer_name[4: len(layer_name)])
                NN_part1.append(nn.MaxPool2d(pooling_size, pooling_size))
                last_layer = 'pool'
                pooling_scale *= pooling_size
            elif 'fc' in layer_name:
                if last_layer != 'fc':
                    pooling_square_length = 32 / pooling_scale
                    view_size = last_layer_output_num * pooling_square_length * pooling_square_length
                    last_layer_output_num = view_size
                fc_size = int(layer_name[2: len(layer_name)])
                tmp_fc = nn.Linear(last_layer_output_num, fc_size)
                tmp_fc.weight.data.normal_(0, 0.01)
                tmp_fc.bias.data.fill_(0)
                NN_part2.append(tmp_fc)
                NN_part2.append(nn.BatchNorm1d(fc_size))
                NN_part2.append(nn.ReLU())
                last_layer = 'fc'
                last_layer_output_num = fc_size
        elif 'batch_size' in item[0]:
            config_net.batch_size = int(item[1])
        elif 'learning_rate' in item[0]:
            config_net.learning_rate = float(item[1])
        elif 'epoch_num' in item[0]:
            config_net.epoch_num = int(item[1])
        elif 'early_stop_threshold' in item[0]:
            config_net.early_stop_threshold = int(item[1])
    NN_part2.append(nn.Linear(last_layer_output_num, 10))
    model = NN(NN_part1, NN_part2, view_size)
    if store_model:
        model_path = os.path.join(argparser.parse_args().model_dir, model_name + ".mod")
        torch.save(model.state_dict(), model_path)
    return config_net, model


def MN_generate(config_files, mother_file):
    '''
    Generate the MotherNet configuration file from the ensemble networks' configuration files
    :param config_files: the ensemble networks' configuration files
    :param mother_file: generated MotherNet configuration file
    '''
    MN_part1 = []
    MN_part2 = []
    MN_config = Config_Net()

    for config_idx, config_file in enumerate(config_files):
        if config_idx == 0:
            config = ConfigParser.ConfigParser()
            config.read(config_file)
            config_items = config.items('hyperparameter')
            block = []
            for item in config_items:
                if 'layer' in item[0]:
                    layer_name = item[1].strip()
                    if 'conv' in layer_name:
                        dash = layer_name.index('-')
                        filter_size = int(layer_name[dash + 1: len(layer_name)])
                        kernel_size = int(layer_name[4: dash])
                        block.append((filter_size, kernel_size))
                    elif 'pool' in layer_name:
                        MN_part1.append(block)
                        block = []
                    elif 'fc' in layer_name:
                        fc_size = int(layer_name[2: len(layer_name)])
                        MN_part2.append(fc_size)
                elif 'batch_size' in item[0]:
                    MN_config.batch_size = int(item[1])
                elif 'learning_rate' in item[0]:
                    MN_config.learning_rate = float(item[1])
                elif 'epoch_num' in item[0]:
                    MN_config.epoch_num = int(item[1])
                elif 'early_stop_threshold' in item[0]:
                    MN_config.early_stop_threshold = int(item[1])
        else:
            config = ConfigParser.ConfigParser()
            config.read(config_file)
            config_items = config.items('hyperparameter')
            block = []
            block_idx = 0
            layer_idx = 0
            update = False
            for item in config_items:
                if 'layer' in item[0]:
                    layer_name = item[1].strip()
                    if 'conv' in layer_name:
                        dash = layer_name.index('-')
                        filter_size = int(layer_name[dash + 1: len(layer_name)])
                        kernel_size = int(layer_name[4: dash])
                        block.append((filter_size, kernel_size))
                    elif 'pool' in layer_name:
                        if block_idx < len(MN_part1):
                            tmp_block = []
                            for conv_idx in range(len(block)):
                                if conv_idx < len(MN_part1[block_idx]):
                                    tmp_block.append((min(block[conv_idx][0], MN_part1[block_idx][conv_idx][0]), min(block[conv_idx][1], MN_part1[block_idx][conv_idx][1])))
                            if tmp_block != MN_part1[block_idx]:
                                MN_part1[block_idx] = tmp_block
                                update = True
                        block = []
                        block_idx += 1
                    elif 'fc' in layer_name:
                        fc_size = int(layer_name[2: len(layer_name)])
                        if layer_idx < len(MN_part2) and MN_part2[layer_idx] != fc_size:
                            MN_part2[layer_idx] = min(MN_part2[layer_idx], fc_size)
                            update = True
                        layer_idx += 1
                elif 'batch_size' in item[0]:
                    if update:
                        MN_config.batch_size = int(item[1])
                elif 'learning_rate' in item[0]:
                    if update:
                        MN_config.learning_rate = float(item[1])
                elif 'epoch_num' in item[0]:
                    if update:
                        MN_config.epoch_num = int(item[1])
                elif 'early_stop_threshold' in item[0]:
                    if update:
                        MN_config.early_stop_threshold = int(item[1])
            if len(MN_part1) > block_idx:
                MN_part1 = MN_part1[:block_idx]
            if len(MN_part2) > layer_idx:
                MN_part2 = MN_part2[:layer_idx]

    with open(mother_file, 'w') as f:
        f.write("[hyperparameter]\n")
        f.write("# Network structure\n")
        f.write("#\n")
        f.write("# Note: Layer structure specifications\n")
        f.write("# should be in ascending order.\n")
        layer_idx = 1
        for block in MN_part1:
            for conv in block:
                f.write("layer%d = conv%d-%d\n" % (layer_idx, conv[1],conv[0]))
                layer_idx += 1
            f.write("layer%d = pool2\n" % layer_idx)
            layer_idx += 1
        for layer in MN_part2:
            f.write("layer%d = fc%d\n" % (layer_idx, layer))
            layer_idx += 1
        f.write("\n# Training\n")
        f.write("batch_size = %d\n" % MN_config.batch_size)
        f.write("learning_rate = " + str(MN_config.learning_rate) + '\n')
        f.write("epoch_num = %d\n" % MN_config.epoch_num)
        f.write("early_stop_threshold = %d\n" % MN_config.early_stop_threshold)


def torch_state_dict_norm(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' == k[:7]:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return  new_state_dict


if __name__ == '__main__':
    # python MN.py -cn ./Children_Config -mn ./Mother_Config/Mother.conf -m ../model -f ../experiment/ -p ../experiment/ -d SVHN -gm -tm -tc -tcm scratch -tcd bagging -lrd 5 2

    argparser = argparse.ArgumentParser()

    argparser.add_argument('-cn', '--children_nets_dir', type=str, help="path of the directory of configuration files of children networks to be ensembled")
    argparser.add_argument('-mn', '--mother_net_path', type=str, help="path of the configuration file of mother network to be generated")
    argparser.add_argument('-m', '--model_dir', type=str, default="./model/", help="path of the directory to store model files")
    argparser.add_argument('-f', '--form', type=str, default="./", help="path of the directory of result form files that store the status of every training and testing epoch")
    argparser.add_argument('-p', '--plot', type=str, default='', help="path of the directory of result plotting")

    argparser.add_argument('-d', '--dataset', type=str, choices=["SVHN", "CIFAR-10", "CIFAR-100", "None"], default="None", help="dataset to use")

    argparser.add_argument('-gm', '--generate_mother_net', action='store_true', default=False, help="choose to generate the MotherNet")
    argparser.add_argument('-tm', '--train_mother_net', action='store_true', default=False, help="choose to train the MotherNet")
    argparser.add_argument('-tc', '--train_children_nets', action='store_true', default=False, help="choose to train the ChildrenNets")
    argparser.add_argument('-tcm', '--train_children_nets_method', type=str, choices=["MN", "scratch"], default="MN", help="method of training the ChildrenNets")
    argparser.add_argument('-tcd', '--train_children_nets_dataset', type=str, choices=["fully", "bagging"], default="bagging", help="method of using dataset when training the ChildrenNets")

    argparser.add_argument('-lrd', '--learning_rate_decay', type=int, nargs=2, default=(1, 1), help="step and rate of learning rate decay method")


    # load the children configuration files
    children_files = []
    children_names = []
    for single_file in os.listdir(argparser.parse_args().children_nets_dir):
        if single_file.endswith(".conf"):
            children_files.append(os.path.join(argparser.parse_args().children_nets_dir, single_file))
            children_names.append(single_file[: -5])

    # generate the mother configuration file
    if argparser.parse_args().generate_mother_net:
        mother_file = argparser.parse_args().mother_net_path
        MN_generate(children_files, mother_file)


    # load the dataset
    if argparser.parse_args().dataset == "SVHN":
        tra_set = datasets.SVHN(root='SVHN', \
                                split='train', \
                                download=True, \
                                transform=transforms.Compose([transforms.ToTensor(), \
                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        tes_set = datasets.SVHN(root='SVHN', \
                                split='test', \
                                download=True, \
                                transform=transforms.Compose([transforms.ToTensor(), \
                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif argparser.parse_args().dataset == "CIFAR-10":
        tra_set = datasets.CIFAR10('CIFAR10_data', \
                                   train=True, \
                                   download=True, \
                                   transform=transforms.Compose([transforms.RandomCrop(32, padding=4), \
                                                                 transforms.RandomHorizontalFlip(), \
                                                                 transforms.ToTensor(), \
                                                                 transforms.Normalize(mean=[0.491, 0.482, 0.447], \
                                                                                      std=[0.247, 0.243, 0.262])]))
        tes_set = datasets.CIFAR10('CIFAR10_data', \
                                   train=False, \
                                   download=True, \
                                   transform=transforms.Compose([transforms.ToTensor(), \
                                                                 transforms.Normalize(mean=[0.491, 0.482, 0.447], \
                                                                                      std=[0.247, 0.243, 0.262])]))
    elif argparser.parse_args().dataset == "CIFAR-100":
        tra_set = datasets.CIFAR100('CIFAR100_data', \
                                    train=True, \
                                    download=True, \
                                    transform=transforms.Compose([transforms.RandomCrop(32, padding=4), \
                                                                  transforms.RandomHorizontalFlip(), \
                                                                  transforms.ToTensor(), \
                                                                  transforms.Normalize(mean=[0.507, 0.487, 0.441], \
                                                                                       std=[0.267, 0.256, 0.276])]))
        tes_set = datasets.CIFAR100('CIFAR100_data', \
                                    train=False, \
                                    download=True, \
                                    transform=transforms.Compose([transforms.ToTensor(), \
                                                                  transforms.Normalize(mean=[0.507, 0.487, 0.441], \
                                                                                       std=[0.267, 0.256, 0.276])]))


    # generate the result form file
    if argparser.parse_args().train_mother_net or argparser.parse_args().train_children_nets:
        form_train = open(os.path.join(argparser.parse_args().form, 'traininig.csv'), 'w')
        form_train_writer = csv.writer(form_train, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        form_acc = open(os.path.join(argparser.parse_args().form, 'acc.csv'), 'w')
        form_acc_writer = csv.writer(form_acc, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        form_time = open(os.path.join(argparser.parse_args().form, 'time.csv'), 'w')
        form_time_writer = csv.writer(form_time, delimiter=' ', quoting=csv.QUOTE_MINIMAL)


    # train the MotherNet
    if argparser.parse_args().train_mother_net:

        mother_file = argparser.parse_args().mother_net_path
        mother_config, mother_model = config_file2model(mother_file, store_model=False)
        mother_name = mother_file.split('/')[-1].split('.')[0]
        mother_model_path = os.path.join(argparser.parse_args().model_dir, mother_name + ".mod")
        form_train_writer.writerow(['train_'+mother_name])

        SL_loader = torch.utils.data.DataLoader(tra_set, \
                                                batch_size=500, \
                                                shuffle=False, \
                                                num_workers=64, \
                                                pin_memory=True)
        fully_train_loader = torch.utils.data.DataLoader(tra_set, \
                                                         batch_size=mother_config.batch_size, \
                                                         shuffle=True, \
                                                         num_workers=64, \
                                                         pin_memory=True)

        test_loader = torch.utils.data.DataLoader(tes_set, \
                                                  batch_size=mother_config.batch_size, \
                                                  num_workers=64, \
                                                  pin_memory=True)

        cuda_enabled = torch.cuda.is_available()
        # cuda_enabled = False
        if cuda_enabled:
            mother_model.cuda()
            mother_model = torch.nn.DataParallel(mother_model, device_ids=range(torch.cuda.device_count()))

        print('\n\n' + mother_name + ': \n')
        print('batch_size = %d' % mother_config.batch_size)
        print('learning_rate = %g' % mother_config.learning_rate)
        print('epoch_num = %d' % mother_config.epoch_num)
        print('early_stop_threshold = %d' % mother_config.early_stop_threshold)

        print('\ntraining:')
        max_acc = 0
        break_counter = 0
        total_time = 0
        for e in range(mother_config.epoch_num):
            if e != 0 and e % argparser.parse_args().learning_rate_decay[0] == 0:
                mother_config.learning_rate *= argparser.parse_args().learning_rate_decay[1]

            print('\nepoch %d: ' % e)
            form_train_writer.writerow(['epoch_%d' % e])
            tmp_time, tmp_acc = train(mother_model, fully_train_loader, mother_config.learning_rate)
            form_train_writer.writerow(['epoch_%d' % e, str(tmp_acc)])

            sys.stdout.flush()
            total_time += tmp_time

            test_acc = test([mother_model], path=False, print_acc=False, super_learn=False)
            if test_acc > max_acc:
                max_acc = test_acc
                break_counter = 0
                torch.save(mother_model.state_dict(), mother_model_path)
            else:
                break_counter += 1

            if break_counter == mother_config.early_stop_threshold:
                print('average epoch time is %d'%(total_time / (e + 1)))
                break

        print('\nMax acc is %g, break_counter is %d' % (max_acc, break_counter))

        print('\ntesting: ')

        test_acc = test([mother_model], path=False, super_learn=False)
        form_acc_writer.writerow([mother_name, str(test_acc)])

        form_time_writer.writerow([mother_name, str(total_time)])

        del mother_model
        del SL_loader
        del fully_train_loader
        del test_loader
        gc.collect()


    # train the ChildrenNets
    if argparser.parse_args().train_children_nets:

        global_total_time = 0
        if argparser.parse_args().train_children_nets_method == 'MN':
            mother_file = argparser.parse_args().mother_net_path
            mother_config, mother_model = config_file2model(mother_file, store_model=False)
            mother_name = mother_file.split('/')[-1].split('.')[0]
            mother_model_path = os.path.join(argparser.parse_args().model_dir, mother_name + ".mod")
            mother_model.load_state_dict(torch_state_dict_norm(torch.load(mother_model_path)))

        children_models_path = []
        for model_idx in range(len(children_names)):

            form_train_writer.writerow(['train_' + children_names[model_idx]])
            children_config, children_model = config_file2model(children_files[model_idx], store_model=False)
            children_model_path = os.path.join(argparser.parse_args().model_dir, children_names[model_idx] + '.mod')
            children_models_path.append(children_model_path)

            if argparser.parse_args().train_children_nets_dataset == 'fully':
                fully_train_loader = torch.utils.data.DataLoader(tra_set, \
                                                                 batch_size=children_config.batch_size, \
                                                                 shuffle=True, \
                                                                 num_workers=64, \
                                                                 pin_memory=True)
                train_loader = fully_train_loader
            else:
                sample_weights = [1] * 50000
                tra_sampler = WeightedRandomSampler(sample_weights, 50000)
                bagging_train_loader = torch.utils.data.DataLoader(tra_set, \
                                                                   batch_size=children_config.batch_size, \
                                                                   sampler=tra_sampler, \
                                                                   num_workers=64, \
                                                                   pin_memory=True)
                train_loader = bagging_train_loader

            SL_loader = torch.utils.data.DataLoader(tra_set, \
                                                    batch_size=500, \
                                                    shuffle=False, \
                                                    num_workers=64, \
                                                    pin_memory=True)
            test_loader = torch.utils.data.DataLoader(tes_set, \
                                                      batch_size=children_config.batch_size, \
                                                      num_workers=64, \
                                                      pin_memory=True)

            if argparser.parse_args().train_children_nets_method == 'MN':
                mother_model.cpu()
                tmp_mother_model = copy.deepcopy(mother_model)
                children_model = function_preserving_transfer(tmp_mother_model, children_model)
                del tmp_mother_model

            cuda_enabled = torch.cuda.is_available()
            # cuda_enabled = False
            if cuda_enabled:
                children_model.cuda()
                children_model = torch.nn.DataParallel(children_model, device_ids=range(torch.cuda.device_count()))

            print('\n\n' + children_names[model_idx] + ': \n')
            print('batch_size = %d' % children_config.batch_size)
            print('learning_rate = %g' % children_config.learning_rate)
            print('epoch_num = %d' % children_config.epoch_num)
            print('early_stop_threshold = %d' % children_config.early_stop_threshold)

            print('\ntraining:')
            max_acc = 0
            break_counter = 0
            total_time = 0
            for e in range(children_config.epoch_num):
                if e != 0 and e % argparser.parse_args().learning_rate_decay[0] == 0:
                    children_config.learning_rate *= argparser.parse_args().learning_rate_decay[1]

                print('\nepoch %d: ' % e)
                form_train_writer.writerow(['epoch_%d' % e])
                tmp_time, tmp_acc = train(children_model, train_loader, children_config.learning_rate)
                form_train_writer.writerow(['epoch_%d'%e, str(tmp_acc)])

                sys.stdout.flush()
                total_time += tmp_time

                test_acc = test([children_model], path=False, print_acc=False, super_learn=False)
                if test_acc > max_acc:
                    max_acc = test_acc
                    break_counter = 0
                    torch.save(children_model.state_dict(), children_model_path)
                else:
                    break_counter += 1

                if break_counter == children_config.early_stop_threshold:
                    print('average epoch time is %d' % (total_time / (e + 1)))
                    break

            print('\nMax acc is %g, break_counter is %d' % (max_acc, break_counter))

            print('\ntesting: ')
            print('\nEnsemble size: %d' % (model_idx + 1))
            test_acc = test(children_models_path[:model_idx+1], children_files[:model_idx+1], path=True, super_learn=False)
            form_acc_writer.writerow([children_names[model_idx], str(test_acc)])

            global_total_time += total_time
            form_time_writer.writerow([children_names[model_idx], str(global_total_time)])

            del children_model
            del train_loader
            if argparser.parse_args().train_children_nets_dataset == 'fully':
                del fully_train_loader
            else:
                del bagging_train_loader
            del SL_loader
            del test_loader
            gc.collect()

        if argparser.parse_args().train_children_nets_method == 'MN':
            del mother_model
            gc.collect()

        print('global_average_time is %d s' % (global_total_time * 195.0 / len(children_names) / 1000))


    # close the result form file
    if argparser.parse_args().train_mother_net or argparser.parse_args().train_children_nets:
        form_train.close()
        form_acc.close()
        form_time.close()


    # plot
    if argparser.parse_args().plot:
        try:
            # trainig accuracy plotting
            with open(os.path.join(argparser.parse_args().form, 'traininig.csv'), 'r') as form:
                plt.figure(1)
                form_reader = csv.reader(form, delimiter=' ')

                train_acc = {}
                for row in form_reader:
                    if 'train' in row[0]:
                        net_name = ''.join(row[0].split('_')[1:])
                        train_acc[net_name] = []
                    if 'epoch' in row[0]:
                        train_acc[net_name].append(float(row[1]) * 100.0)

                mother_epoch = 0
                for net in train_acc:
                    if 'Mother' in net:
                        mother_epoch = len(train_acc[net])

                max_epoch = 0
                for net in train_acc:
                    if 'Mother' in net:
                        plt.plot(range(len(train_acc[net])), train_acc[net], '-x', label = net)
                    else:
                        plt.plot(range(mother_epoch, mother_epoch + len(train_acc[net])), train_acc[net], '-x', label=net)
                        if len(train_acc[net]) > max_epoch:
                            max_epoch = len(train_acc[net])
                plt.title('Traning accurary of each network')
                plt.xlabel('number of epoch')
                plt.ylabel('training accuracy (%)')
                plt.xticks(range(mother_epoch + max_epoch))
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                plt.legend(loc='lower right')

                plt.savefig(os.path.join(argparser.parse_args().plot, 'training.png'))

            # test accuracy plotting
            with open(os.path.join(argparser.parse_args().form, 'acc.csv'), 'r') as form:
                plt.figure(2)
                form_reader = csv.reader(form, delimiter=' ')

                test_acc = []
                for row in form_reader:
                    net_name = row[0]
                    if 'Mother' not in net_name:
                        test_acc.append(float(row[1]) * 100)

                plt.plot(range(1, len(test_acc)+1), test_acc, '-x')
                plt.title('Test accurary of each ensemble network')
                plt.xlabel('size of ensemble')
                plt.ylabel('test accuracy (%)')
                plt.xticks(range(1, len(test_acc)+1))
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

                plt.savefig(os.path.join(argparser.parse_args().plot, 'acc.png'))


            with open(os.path.join(argparser.parse_args().form, 'time.csv'), 'r') as form:
                plt.figure(3)
                form_reader = csv.reader(form, delimiter=' ')

                train_time = []
                time_base = 0
                for row in form_reader:
                    net_name = row[0]
                    if 'Mother' in net_name:
                        time_base = float(row[1])
                    else:
                        if argparser.parse_args().train_children_nets_method != 'scratch':
                            train_time.append((time_base + float(row[1])) / 1000.0)
                        else:
                            train_time.append(float(row[1]) / 1000.0)

                plt.plot(range(1, len(train_time)+1) ,train_time, '-x')
                plt.title('Training time of each ensemble network')
                plt.xlabel('size of ensemble')
                plt.ylabel('training time (s)')
                plt.xticks(range(1, len(train_time)+1))

                plt.savefig(os.path.join(argparser.parse_args().plot, 'time.png'))

        except IOError:
            print('Error: Wrong csv file.')
