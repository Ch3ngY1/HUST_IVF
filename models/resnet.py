# -*- coding: utf-8 -*-
import torch

import numpy as np
import torch.nn as nn

from collections import OrderedDict
from models import Action
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, head):
        self.inplanes = 64
        self._head = head
        self._block = layers
        super(ResNet, self).__init__()

        m = OrderedDict()
        # TODO:
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        if self._head:
            self.group2 = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(512 * block.expansion, num_classes))
                ])
            )
        self.out_channels = 512 * block.expansion
        self.channels_per_layer = (np.array([64, 128, 256, 512]) * block.expansion).tolist()
        self.forward_list = ['group1', 'layer1', 'layer2', 'layer3', 'layer4']

        # self.action1 =

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        # TODO: Add Action
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self._head:
            x = self.group2(x)
        return x


    def froze_bn(self):
        froze_modules = [self.group1, self.layer1, self.layer2,
                         self.layer3, self.layer4]
        self._freeze_bn(froze_modules)

    def _freeze_bn(self, froze_modules):
        for md in froze_modules:
            # print(type(md))
            if isinstance(md, nn.BatchNorm2d):
                md.eval()
                for param in md.parameters():
                    param.requires_grad = False
            elif isinstance(md, nn.Sequential):
                self._freeze_bn(md)
            elif isinstance(md, Bottleneck):
                self._freeze_bn(md.group1)
                if md.downsample is not None:
                    self._freeze_bn(md.downsample)

    def fix_model(self):
        fix_modules = [self.group1, self.layer1]
        for md in fix_modules:
            for name, param in md.named_parameters():
                param.requires_grad = False

    def load_official_state_dict(self, state_dict):
        import re
        from collections import OrderedDict
        own_state_old = self.state_dict()
        own_state = OrderedDict()  # remove all 'group' string
        new_state_dict = OrderedDict()
        for k, v in own_state_old.items():
            k = re.sub('group\d+\.', '', k)
            own_state[k] = v

        for k, v in state_dict.items():
            k = re.sub('model+\.', '', k)
            new_state_dict[k] = v

        for name, param in new_state_dict.items():
            if name not in own_state:
                print(name)
                raise ValueError

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_state_dict(self, state_dict):
        import re
        from collections import OrderedDict
        own_state = self.state_dict()
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = re.sub('model+\.', '', k)
            new_state_dict[k] = v

        for name, param in new_state_dict.items():
            if name not in own_state:
                print(name)
                raise ValueError

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_model(self, model_path, fc_switch=True):
        pre_model = torch.load(model_path)
        if model_path.count('.torch') != 0:
            pre_model_dict = {k: v for k, v in pre_model.items() if ('fc' not in k)}
            pass
        else:
            pre_model = pre_model['state_dict']
            if self._head and fc_switch:
                pre_model_dict = {k: v for k, v in pre_model.items()}
            else:
                pre_model_dict = {k: v for k, v in pre_model.items() if ('fc' not in k)}
        if model_path.count('.torch') != 0:
            self.load_official_state_dict(pre_model_dict)
        else:
            self.load_state_dict(pre_model_dict)

def resnet18(num_classes=1000, init_function=None, head=True):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, head)
    model.apply(init_function)
    return model

def resnet34(num_classes=1000, init_function=None, head=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, head)
    model.apply(init_function)
    return model

def resnet50(num_classes=1000, init_function=None, head=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, head)
    model.apply(init_function)
    return model

def resnet101(num_classes=1000, init_function=None, head=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, head)
    model.apply(init_function)
    return model

def resnet152(num_classes=1000, init_function=None, head=True):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, head)
    model.apply(init_function)
    return model