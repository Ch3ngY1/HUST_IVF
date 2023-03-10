# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from models import resnet
from models import densenet
from models import p3d_model
from models import i3dpt
import os
import math
import torch.nn as nn
from models import Res_plus_p3d_plus_LSTM, Res_plus_p3d_plus_GRU, Res_plus_p3d_plus_ViTCLS, \
    Res_plus_p3d_plus_LSTM_duo,Res_plus_p3d_plus_LSTM_resgumbel, Dense_plus_p3d_plus_LSTM


def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        he(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        he(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def he(param):
    nn.init.kaiming_uniform(param)

def xavier(param):
    nn.init.xavier_uniform_(param)

os.environ['CUDA_VISIBLE_DEVICES']='6'
import torch
from models import Res_plus_p3d_plus_LSTM
bs = 1
input_img = torch.rand([bs, 32, 1, 224, 224]).cuda()
input_pos = torch.rand([bs, 32, 2048]).cuda()
input_ = [input_img, input_pos]

# checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan03_04-37-55_Ada_Ada_del_random_fix_initialized/best_val_acc.pth', map_location=lambda storage, loc: storage)


checkpoint = torch.load(
    '/data2/chengyi/myproject/Savings/save_models/Feb15_07-31-55_Ada_Ada_new-reward/last_checkpoint.pth', )

net = checkpoint['net'].cuda()

# out = net(input_)

flops, params = profile(net, (input_,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f GFLOPs, params: %.2f M' % (flops / 1e9, params / 1000000.0))

# # Model
# print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)
#
# dummy_input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))



