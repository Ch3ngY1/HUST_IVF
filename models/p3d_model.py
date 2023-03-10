from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
import os
# from TPN import Relation_Resoning_Simple
from functools import partial
# from utils.my_loadmodel import weights_init_xavier, weights_init_he, weights_init_normal


__all__ = ['P3D', 'P3D63', 'P3D131', 'P3D199']


def conv_S(in_planes, out_planes, stride=1, padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=1,
                     padding=padding, bias=False)


def conv_T(in_planes, out_planes, stride=1, padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=1,
                     padding=padding, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_s=0, depth_3d=47, ST_struc=('A', 'B', 'C')):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.depth_3d = depth_3d
        self.ST_struc = ST_struc
        self.len_ST = len(self.ST_struc)

        stride_p = stride
        if not self.downsample == None:
            stride_p = (1, 2, 2)
        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            if n_s == self.depth_3d:
                stride_p = 2
            else:
                stride_p = 1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
            self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id = n_s
        self.ST = list(self.ST_struc)[self.id % self.len_ST]
        if self.id < self.depth_3d:
            self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(planes)
            #
            self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0))
            self.bn3 = nn.BatchNorm3d(planes)
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)

        if n_s < self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm3d(planes * 4)
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def ST_B(self, x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x + tmp_x

    def ST_C(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x + tmp_x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        if self.id < self.depth_3d:  # C3D parts:

            if self.ST == 'A':
                out = self.ST_A(out)
            elif self.ST == 'B':
                out = self.ST_B(out)
            elif self.ST == 'C':
                out = self.ST_C(out)
        else:
            out = self.conv_normal(out)  # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Action(nn.Module):
    def __init__(self, in_channels, shift_div=8):
        super(Action, self).__init__()
        # self.n_segment = n_segment
        self.in_channels = in_channels
        self.reduced_channels = self.in_channels // 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
            self.in_channels, self.in_channels,
            kernel_size=3, padding=1, groups=self.in_channels,
            bias=False)
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1  # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right

        if 2 * self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1  # fixed

        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1), bias=False, padding=(1, 1, 1))

        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1, 1),
                                           bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1,
                                         bias=False, padding=1,
                                         groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0))

        # motion excitation
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1, 1),
                                           bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3),
                                         stride=(1, 1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0))
        # print('=> Using ACTION')

    def forward(self, x):
        n_batch, c_original ,nsegment, h_original ,w_original = x.shape
        x = x.transpose(2,1).contiguous()
        x = x.view(n_batch*nsegment, c_original ,h_original ,w_original)
        nt, c, h, w = x.size()

        x_shift = x.view(n_batch, nsegment, c, h, w)
        x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x_shift = x_shift.contiguous().view(n_batch * h * w, c, nsegment)
        x_shift = self.action_shift(x_shift)  # (n_batch*h*w, c, n_segment)
        x_shift = x_shift.view(n_batch, h, w, c, nsegment)
        x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x_shift = x_shift.contiguous().view(nt, c, h, w)

        # 3D convolution: c*T*h*w, spatial temporal excitation
        nt, c, h, w = x_shift.size()
        x_p1 = x_shift.view(n_batch, nsegment, c, h, w).transpose(2, 1).contiguous()
        x_p1 = x_p1.mean(1, keepdim=True)
        x_p1 = self.action_p1_conv1(x_p1)
        x_p1 = x_p1.transpose(2, 1).contiguous().view(nt, 1, h, w)
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x_shift * x_p1 + x_shift

        # 2D convolution: c*T*1*1, channel excitation
        x_p2 = self.avg_pool(x_shift)
        x_p2 = self.action_p2_squeeze(x_p2)
        nt, c, h, w = x_p2.size()
        x_p2 = x_p2.view(n_batch, nsegment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        x_p2 = self.action_p2_conv1(x_p2)
        x_p2 = self.relu(x_p2)
        x_p2 = x_p2.transpose(2, 1).contiguous().view(-1, c, 1, 1)
        x_p2 = self.action_p2_expand(x_p2)
        x_p2 = self.sigmoid(x_p2)
        x_p2 = x_shift * x_p2 + x_shift

        # # 2D convolution: motion excitation
        x3 = self.action_p3_squeeze(x_shift)
        x3 = self.action_p3_bn1(x3)
        nt, c, h, w = x3.size()
        x3_plus0, _ = x3.view(n_batch, nsegment, c, h, w).split([nsegment - 1, 1], dim=1)
        x3_plus1 = self.action_p3_conv1(x3)

        _, x3_plus1 = x3_plus1.view(n_batch, nsegment, c, h, w).split([1, nsegment - 1], dim=1)
        x_p3 = x3_plus1 - x3_plus0
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3 + x_shift

        # out = x_p1 + x_p2 + x_p3
        out = x_p1 + x_p2
        return out.view(n_batch, nsegment, c_original, h_original, w_original).transpose(2,1).contiguous()


def TSM(x, fold_div=8):
    N,C,T,H,W = x.shape
    x_t = x.transpose(1,2)
    out = torch.zeros_like(x_t)
    fold = C // fold_div
    out[:, :-1, :fold] = x_t[:, 1:, :fold]  # shift left
    out[:, 1:, fold: 2 * fold] = x_t[:, :-1, fold: 2 * fold]  # shift right
    out[:, :, 2 * fold:] = x_t[:, :, 2 * fold:]  # not shift

    return out.transpose(1,2)

class P3D(nn.Module):

    def __init__(self, block, layers, modality='RGB', action=False, pool_factor=None,
                 shortcut_type='B', num_classes=400, dropout=0.5, ST_struc=('A', 'B', 'C')):
        if pool_factor is None:
            pool_factor = '1111'
        self.inplanes = 64
        super(P3D, self).__init__()
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
        #                        padding=(3, 3, 3), bias=False)
        self.pool_factor = pool_factor
        self.action = action
        if self.action:
            print('Action Module activated')
        if modality == 'RGB':
            self.input_channel = 3
        elif modality == 'FlowPlusGray':
            self.input_channel = 3 * 5
        elif modality in ['SPECTRUM', 'CANCELLED', 'GRAY']:
            self.input_channel = 1
        elif modality == 'CSP':
            # TODO confirm its shape
            self.input_channel = 8
        else:
            self.input_channel = 10  # for flow
        self.ST_struc = ST_struc

        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                      padding=(0, 3, 3), bias=False)

        self.depth_3d = sum(layers[:3])  # C3D layers are only (res2,res3,res4),  res5 is C2D

        self.bn1 = nn.BatchNorm3d(64)  # bn1 is followed by conv1
        self.cnt = 0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))  # pooling layer for conv1.
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), padding=0,
                                      stride=(2, 1, 1))  # pooling layer for res2, 3, 4.

        self.action_module1 = Action(64)
        self.action_module2 = Action(256)
        self.action_module3 = Action(512)
        self.action_module4 = Action(1024)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        # self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)  # pooling layer for res5.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # some private attribute
        self.input_size = (self.input_channel, 16, 160, 160)  # input of the network
        self.input_mean = [0.485, 0.456, 0.406] if modality == 'RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality == 'RGB' else [np.mean([0.229, 0.224, 0.225])]

        # self.pyramid_low = Relation_Resoning_Simple.Feature_Pyramid_low()
        # self.pyramid_mid = Relation_Resoning_Simple.Feature_Pyramid_Mid()
        # self.pyramid_high = Relation_Resoning_Simple.Feature_Pyramid_High()
        # self.reason = Relation_Resoning_Simple.Resoning(num_class=2)

        # self.pooltpn = nn.AdaptiveAvgPool3d(kernel_size=(7, 7), stride=7)
        self.cm5 = nn.Conv3d(2048, 2048, kernel_size=1)
        self.cm4 = nn.Conv3d(1024,1024,kernel_size=1)
        self.cm3 = nn.Conv3d(512, 512, kernel_size=1)
        self.cm2 = nn.Conv3d(256, 256, kernel_size=1)

        self.cm4up = nn.Conv3d(2048, 1024, kernel_size=1)
        self.cm3up = nn.Conv3d(1024, 512, kernel_size=1)
        self.cm2up = nn.Conv3d(512, 256, kernel_size=1)

        self.cp4 = nn.Conv3d(1024, 1024, kernel_size=1)
        self.cp3 = nn.Conv3d(512, 512,kernel_size=1)
        self.cp2 = nn.Conv3d(256, 256, kernel_size=1)

        # self.classifier=[]
        # for i in range(3):
        #     classifier = nn.Sequential(
        #         self.dropout(),
        #         nn.Linear(512 * block.expansion, num_classes,bias=True)
        #     )
        #     self.classifier.append()
        self.classifier2 = nn.Linear(65536, num_classes, bias=True)
        self.classifier3 = nn.Linear(16384, num_classes, bias=True)
        self.classifier4 = nn.Linear(4096, num_classes, bias=True)
        self.classifier5= nn.Linear(2048, num_classes, bias=True)

    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160  # asume that raw images are resized (340,256).

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p = stride  # especially for downsample branch.

        if self.cnt < self.depth_3d:
            if self.cnt == 0:
                stride_p = 1
            else:
                stride_p = (1, 2, 2)
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(planes * block.expansion)
                    )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_s=self.cnt, depth_3d=self.depth_3d,
                            ST_struc=self.ST_struc))
        self.cnt += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_s=self.cnt, depth_3d=self.depth_3d, ST_struc=self.ST_struc))
            self.cnt += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.pool_factor[0] == '1':
            x = self.maxpool(x)
        else:
            x = -self.maxpool(-x)
        if self.action:
            x = self.action_module1(x)
        # ------------------Part Res2------------------
        if self.pool_factor[1] == '1':
            x2 = self.maxpool_2(self.layer1(x))
        else:
            x2 = -self.maxpool_2(-self.layer1(x))
        if self.action:
            x2 = self.action_module2(x2)
        # ------------------Part Res3------------------
        if self.pool_factor[2] == '1':
            x3 = self.maxpool_2(self.layer2(x2))
        else:
            x3 = -self.maxpool_2(-self.layer2(x2))
        if self.action:
            x3 = self.action_module3(x3)
        # ------------------Part Res4------------------
        if self.pool_factor[3] == '1':
            x4 = self.maxpool_2(self.layer3(x3))
        else:
            x4 = -self.maxpool_2(-self.layer3(x3))
        if self.action:
            x4 = self.action_module4(x4)
        # ------------------Part Res5------------------
        sizes = x4.size()
        x = x4.view(-1, sizes[1], sizes[3], sizes[4])
        x = self.layer4(x)

        # xx = x.view(-1, )
        # sizes = x.size()
        # x5 = x.view(-1,sizes[1],1,sizes[2],sizes[3])

        # M5 = self.cm5(x5)
        # M4 = self.cm4(x4) + F.interpolate(self.cm4up(M5), size=(1,sizes[2]*2,sizes[3]*2), mode='nearest')
        # M3 = self.cm3(x3) + F.interpolate(self.cm3up(M4), scale_factor=2, mode='nearest')
        # M2 = self.cm2(x2) + F.interpolate(self.cm2up(M3), scale_factor=2, mode='nearest')
        #
        # P5 = M5
        # P4 = self.cp4(M4)
        # P3 = self.cp3(M3)
        # P2 = self.cp2(M2)


        # more than 16 frames
        # sizes = x.size()
        # x = x.view(-1, sizes[1], 2, sizes[2], sizes[3])  # Part Res5
        # x = self.maxpool_2(x)
        # x = x.view(-1, sizes[1], sizes[2], sizes[3])  # Part Res5
        # ------------------ Part FC ------------------
        # P2 = F.avg_pool3d(P2, (1, 7, 7))
        # P3 = F.avg_pool3d(P3, (1, 7, 7))
        # P4 = F.avg_pool3d(P4, (1, 7, 7))
        # P5 = F.avg_pool3d(P5, (1, 7, 7))
        #
        # out2 = self.classifier2(P2.view(-1,65536))
        # out3 = self.classifier3(P3.view(-1,16384))
        # out4 = self.classifier4(P4.view(-1,4096))
        # out5 = self.classifier5(P5.view(-1,2048))
        #
        # out = (out2 + out3 + out4 + out5)/4


        x = self.avgpool(x)
        x = x.view(-1, self.fc.in_features)
        out = self.fc(self.dropout(x))

        return out

    def load_state_dict(self, state_dict):
        from torch.utils import model_zoo
        from torch import nn
        import re
        from collections import OrderedDict
        own_state_old = self.state_dict()
        own_state = OrderedDict()  # remove all 'group' string
        new_state = OrderedDict()  # remove all 'group' string
        for k, v in own_state_old.items():
            k = re.sub('group\d+\.', '', k)
            own_state[k] = v

        for k, v in state_dict.items():
            k = re.sub('module\.', '', k)
            new_state[k] = v

        for name, param in new_state.items():
            if name not in own_state or name.count('fc') != 0:
                print(name)
                continue

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

        missing = (set(new_state.keys()) - set(own_state.keys())) | (set(own_state.keys()) - set(new_state.keys()))
        print('missing keys in state_dict: '.format(missing))

    def load_model(self, model_path, fc_switch=True):
        pre_model = torch.load(model_path)
        if model_path.count('CommonWeights') != 0:
            pre_model = pre_model['state_dict']
            pre_model_dict = {k: v for k, v in pre_model.items() if ('fc' not in k)}
            for k, v in pre_model.items():
                if 'conv1_custom.weight' in k:
                    if v.size()[1] != self.input_channel:
                        new_value = torch.mean(v, 1, keepdim=True).expand(-1, self.input_channel, -1, -1, -1)
                        pre_model_dict[k] = new_value
        else:
            pre_model = pre_model['state_dict']
            if fc_switch:
                pre_model_dict = {k: v for k, v in pre_model.items()}
            else:
                pre_model_dict = {k: v for k, v in pre_model.items() if ('fc' not in k)}
        self.load_state_dict(pre_model_dict)


def P3D63(**kwargs):
    """Construct a P3D63 modelbased on a ResNet-50-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def P3D131(**kwargs):
    """Construct a P3D131 model based on a ResNet-101-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def P3D199(num_classes=1000, init_function=None, modality='RGB', action=False, pool_factor=None, **kwargs):
    """construct a P3D199 model based on a ResNet-152-3D model.
    """
    model = P3D(Bottleneck, [3, 8, 36, 3], modality=modality, action=action, pool_factor=pool_factor, num_classes=num_classes, **kwargs)
    model.apply(init_function)
    return model


# custom operation
def get_optim_policies(model=None, modality='RGB', enable_pbn=True):
    '''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn2, and many all bn3.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model == None:
        print('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate = 0.7
    n_fore = int(len(normal_weight) * slow_rate)
    slow_feat = normal_weight[:n_fore]  # finetune slowly.
    slow_bias = normal_bias[:n_fore]
    normal_feat = normal_weight[n_fore:]
    normal_bias = normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "slow_bias"},
        {'params': normal_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "normal_feat"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
    ]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = P3D199(num_classes=2, init_function=weights_init_xavier, action=True)
    print(model)
    model = model.cuda()
    data = torch.autograd.Variable(
        torch.rand(1, 3, 16, 224, 224)).cuda()  # if modality=='Flow', please change the 2nd dimension 3==>2
    out = model(data)
    print(out.size(), out)
