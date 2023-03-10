
import sys
sys.path.append('../')
import numpy as np
import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel, random_fix
from utils.config import cfg
import time
from sklearn import metrics

if __name__ == '__main__':

    args = my_parse.parse_args()

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0
    # last_checkpoint.pth
    # best_val_acc.pth
    checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan29_14-06-24_Ada_Ada_tau2/best_val_acc.pth')
    # print("use pre-trained weight")
    net = checkpoint['net'].cuda()

    test_dataloader = my_dataloader.my_dataloader(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage',
                                                 filename='test_resample.json', args=args, train=False)

    net = net.cuda()


    test_predict_list, test_target_list = [], []
    test_loss_total = 0
    for batch_idx, (blobs, targets) in enumerate(test_dataloader):
        cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
        if args.modality == 'Ada':
            blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
        else:
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)

        with torch.no_grad():
            if args.net == 'Ada':
                pred_list = net(blobs)

                if args.cell_pred:
                    final_pred = pred_list[1]
                    pred = final_pred[-1].data.max(1)[1]
                else:
                    final_pred = pred_list[0]
                    pred = final_pred[-1].data.max(1)[1]
            else:
                cls_preds = net(blobs)

                # TODO: mixup in loss function

                if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                    pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

        test_predict_list += list(pred.cpu().numpy())
        test_target_list += list(targets)


        utils.progress_bar(batch_idx, len(test_dataloader), 'Acc: %.3f'
                           % (100. * metrics.accuracy_score(test_target_list, test_predict_list)))

    print(100. * metrics.accuracy_score(test_target_list, test_predict_list))
    recall = [metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[cls])
     for cls in range(2)]
    print(recall)