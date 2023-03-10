import os
import sys
sys.path.append("../")
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel, random_fix
from utils.config import cfg
import time
from sklearn import metrics
from torchvision.models import densenet201
import torch.nn as nn

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


def temporal():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    # net = Spatial_Model()
    net = densenet201(pretrained=True)

    train_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                                  filename='train.json', args=args, train=True)

    val_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                                  filename='valid.json', args=args, train=False)
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

def spatial_train():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    # net = Spatial_Model()
    net = densenet201(pretrained=True)

    train_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                                  filename='train.json', args=args, train=True)

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
    with open('../../Savings/save_models/skgbt/clf.pickle', 'wb') as f:
        pickle.dump(clf, f)


def spatial_val():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0

    # net = Spatial_Model()
    net = densenet201(pretrained=True)
    # net.froze_bn()

    val_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                                filename='valid.json', args=args, train=False)


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

    import pickle  # pickle模块

    with open('../../Savings/save_models/skgbt/clf.pickle', 'rb') as f:
        clf2 = pickle.load(f)


    print("============================= Validate =============================")


    for batch_idx, (blobs, targets) in enumerate(val_dataloader):
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

            utils.progress_bar(batch_idx, len(val_dataloader))

        # if batch_idx == 10:
        #     break
    data_in_list = data_in_list.reshape(-1, 35*1000)
    # data_in_list = data_in_list.reshape(22, 35*1000)
    y = clf2.decision_function(data_in_list)
    a = clf2.predict_proba(data_in_list)
    b = clf2.predict(data_in_list)
    '''
    xxx = clf2.predict_proba(data_in_list)
    100*(1 - abs(np.argmax(xxx,axis=1) - data_label_list).sum()/330)
    >>> 60.606
    '''
    print(clf2.score(data_in_list, data_label_list))


if __name__ == '__main__':
    # spatial_train()
    spatial_val()