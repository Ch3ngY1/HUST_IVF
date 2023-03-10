import os
import sys
sys.path.append("../")
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel, random_fix
from sklearn.ensemble import GradientBoostingClassifier
from utils.config import cfg
import time
from sklearn import metrics
from torchvision.models import densenet201
import torch.nn as nn
import numpy as np
'''
提取特征-->GBT分类
'''


class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.base = densenet201(pretrained=True)
        self.last = nn.Linear(1000, 5)
        # self.base.classifier = nn.Linear(in_features=1920, out_features=1, bias=True)

    def forward(self, input):
        # input = input.view()
        output = self.base(input)
        output = self.last(output)
        return output


class Spatial_Model(nn.Module):
    def __init__(self):
        super(Spatial_Model, self).__init__()
        self.base = densenet201(pretrained=True)
        self.base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.base.classifier = nn.Linear(in_features=1920, out_features=1, bias=True)
    def forward(self, input):

        # input = input.view()
        output = self.base(input)
        return output

def train():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    # net = Spatial_Model()
    # net = densenet201(pretrained=True)
    net = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan25_09-07-43_Ada_Ada_cellcountRE/best_val_acc.pth')['net']
    # net = nn.Sequential(*list(net.children())[:-1])
    net = list(net.children())[0]
    train_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
                                                                  filename='train_FOR_STEM_FINAL_RE.json', args=args, train=True)

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Model\'s total number of parameters: %.3f M' % (num_params / 1e6))

    parallel_tag = False
    if GPU:
        net = net.cuda()
        if args.gpus is not None:
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
            assert isinstance(args.gpus, list) and len(args.gpus) > 1
            net.parallel(args.gpus)
            parallel_tag = True
        if not args.random_fix:
            cudnn.benchmark = True

    data_in_list = np.array([])
    data_label_list = np.array([])
    net.eval()
    # Data
    for batch_idx, (blobs, targets) in enumerate(train_dataloader):
        with torch.no_grad():
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)
            bs, seg, fr, h ,w = blobs.shape
            blobs = blobs.view(-1, 1, h, w)
            blobs = blobs.repeat(1,3,1,1)

            cls_out = net(blobs)
            cls_out = cls_out.view(bs, seg*fr, 1000)
            cls_out = cls_out.detach().cpu().numpy()
            data_in_list = np.append(data_in_list, cls_out)

            cls_targets = targets
            data_label_list = np.append(data_label_list, cls_targets)

            utils.progress_bar(batch_idx, len(train_dataloader))


    print("============================= reshape... =============================")
    data_in_list = data_in_list.reshape(-1,35*1000)
    print("============================= training GBT... =============================")
    clf = GradientBoostingClassifier().fit(data_in_list, data_label_list)

    import pickle  # pickle模块

    # 保存Model(注:save文件夹要预先建立，否则会报错)
    with open('../../Savings/save_models/skgbt0125/clf.pickle', 'wb') as f:
        pickle.dump(clf, f)

def valid():
    args = my_parse.parse_args()

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    # net = Spatial_Model()
    # net = densenet201(pretrained=True)
    net = densenet201(pretrained=True)

    test_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
                                                                  filename='valid_FOR_STEM_FINAL_RE.json', args=args, train=True)

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Model\'s total number of parameters: %.3f M' % (num_params / 1e6))

    parallel_tag = False
    if GPU:
        net = net.cuda()
        if args.gpus is not None:
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
            assert isinstance(args.gpus, list) and len(args.gpus) > 1
            net.parallel(args.gpus)
            parallel_tag = True
        if not args.random_fix:
            cudnn.benchmark = True

    data_in_list = np.array([])
    data_label_list = np.array([])
    net.eval()
    # Data
    for batch_idx, (blobs, targets) in enumerate(test_dataloader):
        with torch.no_grad():
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)
            bs, seg, fr, h ,w = blobs.shape
            blobs = blobs.view(-1, 1, h, w)
            blobs = blobs.repeat(1,3,1,1)

            cls_out = net(blobs)
            cls_out = cls_out.view(bs, seg*fr, 1000)
            cls_out = cls_out.detach().cpu().numpy()
            data_in_list = np.append(data_in_list, cls_out)

            cls_targets = targets
            data_label_list = np.append(data_label_list, cls_targets)

            utils.progress_bar(batch_idx, len(test_dataloader))


    print("============================= reshape... =============================")
    data_in_list = data_in_list.reshape(-1,35*1000)

    import pickle  # pickle模块
    with open('../../Savings/save_models/skgbt0125/clf.pickle', 'rb') as fr:
        cls = pickle.load(fr)
    print('acc = {}'.format(cls.score(data_in_list, data_label_list)))
    # cls.predict_proba(data_in_list)
    # cls.predict(data_in_list)
if __name__ == '__main__':

    # train()
    valid()