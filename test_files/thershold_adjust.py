'''
对于保存好的模型对验证集进行权重微调后查看结果
因为结果是从acc等出发，因此loss没有作用

用于测试模型validation复现结果
发现问题在data set对于validation的sample是random的

'''
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
    train = False
    args = my_parse.parse_args()



    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0

    # Jan28_07 - 0
    # 9 - 01
    # _Ada_Ada_remake - dataset - no - long
    checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan30_10-29-50_Ada_Ada_onlyLSTM/last_checkpoint.pth')
    # print("use pre-trained weight")
    net = checkpoint['net'].cuda()

    val_dataloader = my_dataloader.my_dataloader(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage',
                                                 filename='test_resample.json', args=args, train=False)


    net = net.cuda()


    test_predict_list, test_target_list, pred_list_for_adjust, test_predict_prob = [], [], [], []
    net.eval()
    val_loss_total = 0
    for batch_idx, (blobs, targets) in enumerate(val_dataloader):
        cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)

        blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)


        with torch.no_grad():
            if args.net == 'Ada':
                pred_list = net(blobs)

                final_pred = pred_list[1]
                # if pred_list == []:
                #     pred_list_for_adjust = list(final_pred[-1].cpu().numpy())
                # else:
                pred_list_for_adjust += list(final_pred[-1].cpu().numpy())
                pred = final_pred[-1].data.max(1)[1]
                prob = final_pred[-1].data

        test_predict_list += list(pred.cpu().numpy())
        test_target_list += list(targets)
        test_predict_prob += list(prob.cpu().numpy())

        utils.progress_bar(batch_idx, len(val_dataloader), 'Acc: %.3f'
                           % (100. * metrics.accuracy_score(test_target_list, test_predict_list)))

    start = 0.1
    step = 0.005
    acc_list = []
    recall_list = []
    for i in np.arange(-start,start,step):
        adjust_value = torch.tensor([i,0]).repeat(330,1)
        adjusted_pred = torch.tensor(pred_list_for_adjust) + adjust_value
        test_predict_list_adjusted = [torch.tensor(pred_each).data.max(dim=0)[1] for pred_each in adjusted_pred]
        acc_list.append(100. * metrics.accuracy_score(test_target_list, test_predict_list_adjusted))
        recall_list.append([metrics.recall_score(test_target_list, test_predict_list_adjusted, average='macro', labels=[cls])
                    for cls in range(2)])

    print(100. * metrics.accuracy_score(test_target_list, test_predict_list))
    recall = [metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[cls])
     for cls in range(2)]
    print(recall)

    target_list_onehot = np.eye(2)[test_target_list]
    auc = [metrics.roc_auc_score(target_list_onehot, test_predict_prob)]

    print(auc)
    test_predict_prob = torch.nn.functional.softmax(torch.tensor(test_predict_prob)).numpy()
    # fpr, tpr, thresholds = metrics.roc_curve(test_target_list, test_predict_prob)
    fpr, tpr, thresholds = metrics.roc_curve(test_target_list, test_predict_prob[:, 1])
    print(fpr)
    print(tpr)
    print(thresholds)
    # print(val_loss_total)