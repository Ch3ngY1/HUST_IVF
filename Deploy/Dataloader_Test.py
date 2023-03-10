import sys
sys.path.append('/data2/chengyi/myproject/SourceCode/')
from utils import my_parse, my_dataloader, utils

# \
#     my_loadmodel, my_loss, , my_optimizer, Log, savemodel, random_fix
# from utils.config import cfg
# import time
from sklearn import metrics
import os
import torch.utils.data as data
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import json
import zipfile

from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request,jsonify,make_response
import requests


from config_submit import configs

import torchvision
from PIL import Image, ImageOps


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, num_segments, size, div=True, new_length=1):
        self.div = div
        self.seg = num_segments
        self.size = size
        self.new_length = new_length

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            seg, _, _ = img.shape
            img = img.view(1, seg, self.size, self.size)

        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class AdaDataSet_(data.Dataset):
    def __init__(self, video_name, transform=None):

        self.transform = transform

        cap = cv2.VideoCapture(video_name)
        num = 1
        img_list = []
        while True:
            success, data = cap.read()
            if not success:
                break

            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = Image.fromarray(data)
            img_list.append(data)
            # im = Image.fromarray(data)  # 重建图像
            # im.save('C:/Users/Taozi/Desktop/2019.04.30/' + str(num) + ".jpg")  # 保存当前帧的静态图像
            # num = num + 1
            # print(num)
        cap.release()


        # img_list = os.listdir(root_path)
        # img_list = [int(x.split('.')[0]) for x in img_list]
        # img_list.sort()
        # img_list = [str(x)+'.jpg' for x in img_list]
        self.sample = img_list
        self.sample_interval = 1
        self.num_segments = 32

    def _get_val_indices(self, num_segments=32):

        num_frame = len(self.sample)
        if num_frame > 750:
            num_frame = 750

        num_frame = num_frame - 50

        if num_frame > num_segments + self.sample_interval - 1:
            tick = (num_frame - self.sample_interval + 1) / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))

        return offsets + 50

    def __getitem__(self, index):
        segment_indices = self._get_val_indices(self.num_segments)
        return self.get(segment_indices)

    def get(self, indices):
        images = list()
        indexes = list()
        for seg_ind in indices:
            p = int(seg_ind)

            # seg_imgs = [Image.open(os.path.join(self.root_path, self.sample[p])).convert('L')]
            seg_imgs = [self.sample[p]]


            images.extend(seg_imgs)
            indexes.append(p)
            if p < len(self.sample):
                p += self.sample_interval
        process_data = self.transform(images)
        positional_encoding = self.positional_encoding(indexes)
        return process_data, positional_encoding

    def __len__(self):
        return 1

    def positional_encoding(self, indexes, max_len=2048):
        import math
        dim = 850
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.transpose(1, 0)
        pe = torch.stack([pe[x] for x in indexes], dim=0)
        return pe



    def classify_collate(self, batch):


        if isinstance(batch[0][1], torch.Tensor):
            positional_encoding = torch.stack([x[1] for x in batch])
        elif isinstance(batch[0][1], list):
            positional_encoding = [x[1] for x in batch]
        else:
            positional_encoding = None
        imgs = [x[0] for x in batch]

        return [torch.stack(imgs, 0).transpose(2, 1), positional_encoding]


augmentation = torchvision.transforms.Compose([GroupScale(int(256)),
                           GroupCenterCrop(224)])

transformer = torchvision.transforms.Compose([
    augmentation,
    Stack(roll=False),
    ToTorchFormatTensor(32, 224, div=True),
])

def my_dataloader(root):
    transform = transformer

    dataset = AdaDataSet_(video_name=root, transform=transform)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    GPU = 1
    checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/May11_04-46-09_Ada_Ada_newest/best_t_acc.pth')
    net = checkpoint['net'].cuda()
    # test_dataloader = my_dataloader(root='/data2/chengyi/myproject/SourceCode/Deploy/test/zip/standard_489.avi')
    net.eval()
    pos = []
    neg = []
    for avi in os.listdir('/data/mxj/IVF_HUST/first_travel_VideoImage/Videos'):
        test_dataloader = my_dataloader(root='/data/mxj/IVF_HUST/first_travel_VideoImage/Videos/'+avi)
        for batch_idx, (blobs) in enumerate(test_dataloader):
            blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
            with torch.no_grad():
                pred_list = net(blobs)
                final_pred = pred_list[0]
                pred_origin = final_pred[-1].data
                pred = pred_origin.max(1)[1]
            print(pred.cpu().item())
            if pred.cpu().item() == 0:
                neg.append(avi)
            else:
                pos.append(avi)
    print(len(neg))
    print(len(pos))




