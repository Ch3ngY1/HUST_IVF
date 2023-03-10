import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel
from utils.config import cfg
import time
from sklearn import metrics
import torch.nn as nn
checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan03_04-37-55_Ada_Ada_del_random_fix_initialized/best_val_acc.pth')
net = checkpoint['net'].lstm.cuda()
h_out0 = []
h_out1 = []
c_out0 = []
c_out1 = []
testin = torch.zeros([1,32,2048]).cuda()
out0 = net(testin)
for i in range(32):
    testin = torch.zeros([1,32,2048]).cuda()
    testin[:,i,:] = torch.ones([1,1,2048]).cuda()
    out = net(testin)
    hidden, cell, utility = out
    h_out0.append(hidden[-1][0][0].item())
    h_out1.append(hidden[-1][0][1].item())
    c_out0.append(cell[-1][0][0].item())
    c_out1.append(cell[-1][0][1].item())
    # print(hidden[-1])
    # print(cell[-1])
print(h_out0)
print(h_out1)
print(c_out0)
print(c_out1)