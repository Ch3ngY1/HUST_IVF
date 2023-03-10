# encoding: utf-8
import os
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
import sys
sys.path.append("../")



class Temporal_Model(nn.Module):
    def __init__(self):
        super(Temporal_Model, self).__init__()
        self.embedding_indice = nn.Embedding(num_embeddings=900, embedding_dim=256)
        self.embedding_cell = nn.Embedding(num_embeddings=5, embedding_dim=256)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256)
        self.drop = nn.Dropout(p=0.5)
        self.ac = nn.Sigmoid()
        self.fc = nn.Linear(256, 2)
        # self.base.classifier = nn.Linear(in_features=1920, out_features=1, bias=True)

    def forward(self, input):
        # [cells, imgs]
        cells, imgs = input
        cells = self.embedding_cell(cells)
        # imgs = imgs-imgs.min()
        imgs = self.embedding_indice(imgs)
        data = torch.cat([imgs, cells], dim=2)
        # x = self.embedding(input)
        x = self.lstm(data)
        x = x[0][:,-1,:]
        x = self.drop(x)
        x = self.fc(x)
        x = self.ac(x)
        return x


def make_readable_dataset(setname):
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    # net = Spatial_Model()
    checkpoint = torch.load('../../Savings/save_models/Jan21_04-47-35_Ada_Ada_cell_count/best_val_acc.pth')
    net = checkpoint['net']

    train_dataloader = my_dataloader.my_dataloader_res_comparison(
        root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
        filename='{}_FOR_STEM_FINAL_RE.json'.format(setname), args=args, train=True, temporal=True)
    json_file_out = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/final/{}_PNF_TEMPORAL_NEW.json'.format(
        setname)
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
    # 数据集为：[name]:{label:xx, cellcount:[xxxx], img_list:[xxxx]}
    out_dir = {}
    net.eval()
    # Data
    for batch_idx, (blobs, targets, videoname, indices) in enumerate(train_dataloader):
        with torch.no_grad():
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)
            # bs, seg, fr, h ,w = blobs.shape
            bs, c, num, h, w = blobs.shape
            blobs = blobs.view(-1, 1, h, w)
            blobs = blobs.repeat(1, 3, 1, 1)

            cls_out = net(blobs)
            # cls_out = cls_out.view(bs, seg*fr, 1000)
            # cls_out = cls_out.detach().cpu().numpy()
            # pred = cls_out.data.max(1)[1].cpu().numpy()
            # data_in_list = np.append(data_in_list, cls_out)
            out_dir[videoname[0]] = {}
            out_dir[videoname[0]]['cell_count'] = cls_out.data.max(1)[1].cpu().numpy().tolist()
            out_dir[videoname[0]]['imgs'] = indices
            out_dir[videoname[0]]['targets'] = targets.tolist()

            # cls_targets = targets
            # data_label_list = np.append(data_label_list, cls_targets)

            utils.progress_bar(batch_idx, len(train_dataloader))
    my_out = json.dumps(out_dir)
    f2 = open(json_file_out, 'w')
    f2.write(my_out)
    f2.close()


def train():
    args = my_parse.parse_args()

    this_train_tag = datetime.now().strftime('%b%d_%H-%M-%S_') + args.net + '_' + args.modality + '_' + args.tag
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)
    GPU = 1

    net = Temporal_Model()
    # TODO: 用valid试试看
    train_dataloader = my_dataloader.my_dataloader_res_comparison(
        root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
        filename='train_PNF_TEMPORAL_NEW.json', args=args, train=True, temporal_training=True)
    val_dataloader = my_dataloader.my_dataloader_res_comparison(
        root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
        filename='valid_PNF_TEMPORAL_NEW.json', args=args, train=False, temporal_training=True)

    parallel_tag = False
    if GPU:
        net = net.cuda()
        cudnn.benchmark = True

    loss = nn.CrossEntropyLoss()
    optimizer = my_optimizer.init_optimizer(net=net, args=args)
    net.train()
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

        train_predict_list, train_target_list = [], []
        net.train()
        train_loss_total = 0
        # Data
        for batch_idx, (blobs, targets, imgs) in enumerate(train_dataloader):

            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            imgs = utils.label_to_variable(imgs, gpu=GPU, volatile=False)
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)

            cls_preds = net([blobs, imgs])
            train_loss_strategy = loss(cls_preds, cls_targets)
            # TODO: mixup in loss function

            if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
            else:
                pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            train_predict_list += list(pred.cpu().numpy())
            train_target_list += list(targets)

            optimizer.zero_grad()
            train_loss = train_loss_strategy
            train_loss_total = train_loss_total + train_loss
            train_loss.backward()
            optimizer.step()

            utils.progress_bar(batch_idx, len(train_dataloader), 'Loss: %.5f | Acc: %.3f'
                               % (train_loss_total / (batch_idx + 1),
                                  100. * metrics.accuracy_score(train_target_list, train_predict_list)))
            # break
        '''
        直接在writer scalar中定义记录部分
        (self, loss, lr, data, n_iter, tag=None):
        '''
        tensorboard_writer.write_scalars(train_loss_total / len(train_dataloader),
                                         lr,
                                         [train_target_list, train_predict_list],
                                         n_iter=epoch,
                                         tag='train')

        savemodel.save_model(net, this_train_tag, 'last_checkpoint.pth', epoch, lr, parallel_tag)
        # =========================================== validation ===========================================
        val_predict_list, val_target_list = [], []
        net.eval()
        val_loss_total = 0
        for batch_idx, (blobs, targets, imgs) in enumerate(val_dataloader):
            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)

            blobs = utils.img_data_to_variable(blobs, gpu=GPU)

            with torch.no_grad():

                cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
                imgs = utils.label_to_variable(imgs, gpu=GPU, volatile=False)
                blobs = utils.img_data_to_variable(blobs, gpu=GPU)

                cls_preds = net([blobs, imgs])
                val_loss_strategy = loss(cls_preds, cls_targets)
                # TODO: mixup in loss function

                if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                    pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            val_predict_list += list(pred.cpu().numpy())
            val_target_list += list(targets)

            val_loss = val_loss_strategy
            val_loss_total = val_loss_total + val_loss

            utils.progress_bar(batch_idx, len(val_dataloader), 'Loss: %.5f | Acc: %.3f | BestAcc: %.3f'
                               % (val_loss_total / (batch_idx + 1),
                                  100. * metrics.accuracy_score(val_target_list, val_predict_list), best_val_acc))

        tensorboard_writer.write_scalars(train_loss_total / len(val_dataloader),
                                         lr,
                                         [val_target_list, val_predict_list],
                                         n_iter=epoch,
                                         tag='validation')

        if metrics.accuracy_score(val_target_list, val_predict_list) > best_val_acc:
            best_val_acc = metrics.accuracy_score(val_target_list, val_predict_list)
            print('Saving best model, acc:{}'.format(best_val_acc))
            savemodel.save_model(net, this_train_tag, 'best_val_acc.pth',
                                 epoch, lr, parallel_tag, val_acc=best_val_acc)

    print('best validation acc = {}'.format(best_val_acc))

    print('Savings are saved at {}'.format(this_train_tag))


def valid():
    args = my_parse.parse_args()

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    # net = Spatial_Model()
    # net = densenet201(pretrained=True)
    net = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan21_04-47-35_Ada_Ada_cell_count/best_val_acc.pth')[
        'net']

    test_dataloader = my_dataloader.my_dataloader_res_comparison(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                                 filename='test.json', args=args, train=True)

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
            bs, seg, fr, h, w = blobs.shape
            blobs = blobs.view(-1, 1, h, w)
            blobs = blobs.repeat(1, 3, 1, 1)

            cls_out = net(blobs)
            cls_out = cls_out.view(bs, seg * fr, 1000)
            cls_out = cls_out.detach().cpu().numpy()
            data_in_list = np.append(data_in_list, cls_out)

            cls_targets = targets
            data_label_list = np.append(data_label_list, cls_targets)

            utils.progress_bar(batch_idx, len(test_dataloader))

    # print("============================= reshape... =============================")
    # data_in_list = data_in_list.reshape(-1,35*1000)

    data_in_list = data_in_list.reshape(-1, 35 * 1000)

    import pickle  # pickle模块
    with open('../../Savings/save_models/skgbt0124/clf.pickle', 'rb') as fr:
        cls = pickle.load(fr)
    cls.score(data_in_list, data_label_list)
    cls.predict_proba(data_in_list)
    cls.predict(data_in_list)


if __name__ == '__main__':
    # setname = 'valid'
    # print('================= {} ===================='.format(setname))
    # make_readable_dataset(setname)
    train()
