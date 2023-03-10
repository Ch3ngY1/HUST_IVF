import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
data = torch.rand([64,16,2048]).cuda()
net = nn.LSTM(input_size=2048,hidden_size=1024).cuda()
with autocast():
    out = net(data)
    print(out)
    print(out[-1].dtype)