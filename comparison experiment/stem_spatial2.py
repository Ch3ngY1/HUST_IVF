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
import torch.nn.functional as F
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
        self.fc1 = nn.Linear(in_features=1920*35, out_features=2, bias=True)
        self.fc2 = nn.Linear(in_features=1920, out_features=2, bias=True)

    def forward(self, input):
        # input.shape = batch * 7 * 5 * 224 * 224
        bs, f, p, h, w = input.shape
        input = input.view(bs*f*p,1,h,w)
        # input = input.view()
        feature = self.base.features(input)
        feature = F.relu(feature, inplace=True)
        feature = F.avg_pool2d(feature, kernel_size=7, stride=1)
        feature = feature.view(bs,-1)
        output = self.fc1(feature)

        # feature = feature.view(bs, 35, -1)
        # feature = feature.mean(dim=1)
        # output = self.fc2(feature)
        # input.shape = (batch * 7 * 5) * 224 * 224
        # 1. 直接cat在一起分类
        # 2. avg以后分类
        return output

def train():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    this_train_tag = 'spatial_direct_DenseNet' + '_' + args.tag
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)

    net = Spatial_Model()
    # net = densenet201(pretrained=True)
    # net = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan25_09-07-43_Ada_Ada_cellcountRE/best_val_acc.pth')['net']
    # net = nn.Sequential(*list(net.children())[:-1])
    # net = list(net.children())[0]
    train_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
                                                                  filename='train_FOR_STEM_FINAL_RE.json', args=args, train=True)
    val_dataloader = my_dataloader.my_dataloader_res_comparison(
        root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
        filename='valid_FOR_STEM_FINAL_RE.json', args=args, train=False)
    optimizer = my_optimizer.init_optimizer(net=net, args=args)
    loss = nn.CrossEntropyLoss()

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

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        if not args.learning_rate_decay:
            lr = args.lr
        else:
            if epoch == 0:
                lr = args.lr
            elif epoch % args.learning_rate_decay == 0 and epoch != 0:
                lr = lr * 0.3
                my_optimizer.adjust_learning_rate(optimizer, lr)
        train_target_list, train_predict_list = [], []
        train_loss_total = 0.0
        net.train()
        # Data
        for batch_idx, (blobs, targets) in enumerate(train_dataloader):
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)
            cls_targets = utils.label_to_variable(targets, gpu=GPU)

            cls_preds = net(blobs)
            train_loss = loss(cls_preds, cls_targets)

            pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            train_predict_list += list(pred.cpu().numpy())
            train_target_list += list(targets)

            optimizer.zero_grad()
            train_loss_total = train_loss_total + train_loss
            train_loss.backward()
            optimizer.step()

            utils.progress_bar(batch_idx, len(train_dataloader), 'Epoch: %d | Lr: %.5f | Loss: %.5f | Acc: %.3f'
                               % (epoch, lr, (train_loss_total / (batch_idx + 1)),
                                  100. * metrics.accuracy_score(train_target_list, train_predict_list)))

        tensorboard_writer.write_scalars(train_loss_total / len(train_dataloader),
                                         lr,
                                         [train_target_list, train_predict_list],
                                         n_iter=epoch,
                                         tag='train')

        savemodel.save_model(net, this_train_tag, 'last_checkpoint.pth', epoch, lr, parallel_tag)

        net.eval()
        # Data
        val_predict_list, val_target_list, val_predict_prob = [], [], []
        val_loss_total = 0.0
        for batch_idx, (blobs, targets) in enumerate(val_dataloader):
            with torch.no_grad():
                blobs = utils.img_data_to_variable(blobs, gpu=GPU)
                cls_targets = utils.label_to_variable(targets, gpu=GPU)

                cls_preds = net(blobs)
                val_loss = loss(cls_preds, cls_targets)

                pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

                val_predict_list += list(pred.cpu().numpy())
                val_target_list += list(targets)
                val_predict_prob += list(cls_preds.cpu().numpy())

                val_loss_total = val_loss_total + val_loss

            utils.progress_bar(batch_idx, len(val_dataloader), 'Epoch: %d | Lr: %.5f | Loss: %.5f | Acc: %.3f'
                               % (epoch, lr, (val_loss_total / (batch_idx + 1)),
                                  100. * metrics.accuracy_score(val_target_list, val_predict_list)))

        tensorboard_writer.write_scalars(train_loss_total / len(val_dataloader),
                                         lr,
                                         [val_target_list, val_predict_list, val_predict_prob],
                                         n_iter=epoch,
                                         tag='validation', auc=True)

        if metrics.accuracy_score(val_target_list, val_predict_list) > best_val_acc:
            best_val_acc = metrics.accuracy_score(val_target_list, val_predict_list)
            print('Saving best model, acc:{}'.format(best_val_acc))
            savemodel.save_model(net, this_train_tag, 'best_val_acc.pth',
                                 epoch, lr, parallel_tag, val_acc=best_val_acc)

def valid():
    args = my_parse.parse_args()

    GPU = 1

    # net = Spatial_Model()
    # net = densenet201(pretrained=True)
    net = torch.load('/data2/chengyi/myproject/Savings/save_models/temporal_direct_DenseNet_concat/last_checkpoint.pth')['net']

    val_dataloader = my_dataloader.my_dataloader_res_comparison(
        root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
        filename='valid_FOR_STEM_FINAL_RE.json', args=args, train=False)


    if GPU:
        net = net.cuda()

    net.eval()
    # Data
    val_predict_list, val_target_list, val_predict_prob = [], [], []

    for batch_idx, (blobs, targets) in enumerate(val_dataloader):
        with torch.no_grad():
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)
            cls_targets = utils.label_to_variable(targets, gpu=GPU)

            cls_preds = net(blobs)

            pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            val_predict_list += list(pred.cpu().numpy())
            val_target_list += list(targets)
            val_predict_prob += list(cls_preds.cpu().numpy())



    print(metrics.accuracy_score(val_target_list, val_predict_list))
if __name__ == '__main__':

    train()
    # valid()