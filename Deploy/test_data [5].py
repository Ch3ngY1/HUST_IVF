# -*- coding:utf-8 -*-
import sys

sys.path.append('/data2/chengyi/myproject/SourceCode')
import os
import torch.utils.data as data
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
import requests
from config_submit import configs
import torchvision
from PIL import Image
import torch
from utils import utils


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


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


class AdaDataSet_(data.Dataset):
    def __init__(self, video_name, transform=None):

        self.transform = transform
        print('load video')
        print('video name=', video_name)
        print(os.path.exists(video_name))
        cap = cv2.VideoCapture(video_name)
        num = 1
        img_list = []

        i = 0

        print('start make frames')

        print(cap)
        while True:
            success, data = cap.read()
            if not success:
                break

            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = Image.fromarray(data)
            img_list.append(data)
            num = num + 1

        cap.release()

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
        out = self.get(segment_indices)
        print(out[0])
        return out

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


executor = ThreadPoolExecutor()
app = Flask(__name__)


@app.route('/test', methods=['GET', 'POST'])
def handle_param():
    # print(request)
    if request.method == 'POST':
        print('1')
        try:
            #   1
            # if True:
            print('2')
            print(request.files.keys())
            file = request.files['file']
            print(file)
            # json_result = request.files['result']
            # request.files['result']
            json_result = None
            send = 'ok'

            print('3')
            file_name = file.filename
            path = "./test/zip/" + file_name
            print('path=', path)
            file.save(path)
            print(path)
            executor.submit(second_handle_param, path, file_name, json_result)
            # print("11")
        except:
            send = "false"

    else:
        return "GET not supported, use POST instead"

    return send


def second_handle_param(path, split_dcm_name, json_result):
    import os
    import torch

    img_url = path
    print(configs['absolute_path'])
    # save_dcm_dir = configs['absolute_path'] + "/test/original/"+split_dcm_name
    # print('save=', save_dcm_dir)
    # if not os.path.exists(save_dcm_dir):
    #     os.mkdir(save_dcm_dir)

    # if split_dcm_name[-1]=='p':
    #     unzip_file(img_url, save_dcm_dir)
    # print('================================')
    print('dcm_name=', split_dcm_name)

    checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/May11_04-46-09_Ada_Ada_newest/last_checkpoint.pth')
    net = checkpoint['net'].cuda()
    print('ckpt already loaded')

    net.eval()
    print('dataloader loading')
    test_dataloader = my_dataloader(root='/data2/chengyi/myproject/SourceCode/Deploy/test/zip/' + split_dcm_name)
    print('dataloader loading finished')

    net.eval()
    for batch_idx, (blobs) in enumerate(test_dataloader):
        blobs[0] = utils.img_data_to_variable(blobs[0], gpu=1)
        with torch.no_grad():
            pred_list = net(blobs)
            final_pred = pred_list[0]
            pred_origin = final_pred[-1].data
            pred = pred_origin.max(1)[1]
        print('?')
        # print(pred_list[0].shape)
    result = pred.cpu().item()
    import torch.nn as nn
    print(pred_origin)
    pred_origin = torch.softmax(pred_list[0][-1].data, dim=1)
    print(pred_origin)
    print(pred_origin[0][1] - pred_origin[0][0])
    if pred_origin[0][1] - pred_origin[0][0] > 0.05:
        input_result = "阳性"
    else:
        input_result = "阴性"


    print(split_dcm_name)

    # result = pred_list[0][-1].data.max(1)[1]
    # print(pred_origin)
    #
    # if result == 0:
    #     input_result = "阴性"
    # elif result == 1:
    #     input_result = "阳性"
    # else:
    #     raise ValueError
    # print(input_result)
    # input_result = '阳性'

    label = [{"templateType": 1,
              "type": "1",
              "typename": "病例分类",
              "value": ["阴性", "阳性"],
              "input": input_result}]

    modalNo = split_dcm_name.split('.')[0]

    json_info = {
        "modalNo": modalNo,
        "data": [],
        "label": label
    }

    head = {"Content-Type": "application/json; charset=UTF-8", 'Connection': 'close'}
    print('===========================start json info===========================')
    print(json_info)
    print('===========================end json info===========================')

    data = json.dumps(json_info)
    r = requests.post(url='http://192.168.0.247:8581/prod-api/orthanc/ai/callback',
                      data=data, headers=head)
    print(r.text)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=9991,
        debug=True
    )
