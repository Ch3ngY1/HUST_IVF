import os
import json
import sys

sys.path.append('../')
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel, random_fix
from utils.config import cfg
import time
# from models import Res_plus_p3d_plus_LSTM_base
from sklearn import metrics

# from torch import torch.loa

'''
瞎猜acc=0.62
暂时将我们的模型命名为Ada
'''
if __name__ == '__main__':

    args = my_parse.parse_args()

    GPU = 1

    # checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Feb09_05-38-11_Ada_Ada_cpcr/best_t_acc.pth')
    checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Feb15_14-09-33_Ada_Ada_new-reward-utility/regular_25.pth')
    net = my_loadmodel.loadmodel(args)
    net.load_model(args.weight)
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    # net = checkpoint['net'].cuda()

    val_dataloader = my_dataloader.my_dataloader(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage',
                                                 filename='test_resample.json', args=args, train=False)

    if GPU:
        net = net.cuda()

    import numpy as np
    chosen_total = 0
    predict_list, target_list, predict_prob = [], [], []
    net.eval()
    val_loss_total = 0
    for batch_idx, (blobs, targets) in enumerate(val_dataloader):
        with torch.no_grad():
            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
            if args.net == 'Ada':
                pred_list = net(blobs)
                _, _, _, use = pred_list
                theshold = 0.75
                for i in range(3):
                    m = torch.stack(use).transpose(dim0=1, dim1=0)[i, :, 1]
                    index = torch.where(m > theshold)
                    # num = m[m>theshold]
                    chosen_total += len(torch.nonzero(torch.nn.ReLU()(m-theshold)))
                # TODO: 可以把重复的减少
                if args.cell_pred:
                    final_pred = pred_list[1]
                    if args.adamode == 'bi':
                        final_pred = pred_list[1] + pred_list[4]
                    pred = final_pred[-1].data.max(1)[1]
                else:
                    final_pred = pred_list[0]
                    if args.adamode == 'bi':
                        final_pred = pred_list[0] + pred_list[3]
                    pred = final_pred[-1].data.max(1)[1]
                prob = final_pred[-1].data
            else:
                cls_preds = net(blobs)

                if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                    pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

        # predict_prob += list(prob.cpu().numpy())
        # predict_list += list(pred.cpu().numpy())
        # target_list += list(targets)

        utils.progress_bar(batch_idx, len(val_dataloader), 'Acc: %.3f'
                           % (100. * metrics.accuracy_score(target_list, predict_list)))

    print(metrics.accuracy_score(target_list, predict_list))

    print(metrics.recall_score(target_list, predict_list, average='macro', labels=[0]))

    print(metrics.recall_score(target_list, predict_list, average='macro', labels=[1]))

    # print(metrics.roc_auc_score(target_list_onehot, pred_prob))
    print('Avg. chosen frame = {}'.format(chosen_total / 330))
#     fpr, tpr, thersholds = metrics.roc_curve(target_list, np.array([x[1] for x in predict_prob]))
#     # if abs(each_weight-0.5) < 0.01:
#     #     print(auc[-1])
#     #     print(recall[-1])
#     #     print(acc[-1])
#     #     print(f1[-1])
#
# #
# # print('acc={}; recall={}; f1={}; auc={}'.format(best_acc, best_recall, best_f1, best_auc))
#
# print(fpr)
# print(tpr)
# print(metrics.auc(fpr, tpr))
# print('?')
# print('Avg. chosen frame = {}'.format(chosen_total/330))
