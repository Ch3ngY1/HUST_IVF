import os
# import json
# from datetime import datetime
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, utils

# \
#     my_loadmodel, my_loss, , my_optimizer, Log, savemodel, random_fix
# from utils.config import cfg
# import time
from sklearn import metrics

'''
瞎猜acc=0.62
暂时将我们的模型命名为Ada
'''
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    args = my_parse.parse_args()

    GPU = 1

    # 加载模型和权重
    checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/May11_04-46-09_Ada_Ada_newest/best_t_acc.pth')
    net = checkpoint['net'].cuda()
    # 加载数据集
    test_dataloader = my_dataloader.my_dataloader(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage',
                                                  filename='test_resample.json', args=args, train=False)

    num_params = 0


    test_predict_list, test_target_list = [], []
    net.eval()
    val_loss_total = 0
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
                    if args.adamode == 'bi':
                        final_pred = pred_list[1] + pred_list[4]

                    pred_origin = final_pred[-1].data

                    pred = pred_origin.max(1)[1]
                else:
                    final_pred = pred_list[0]
                    if args.adamode == 'bi':
                        final_pred = pred_list[0] + pred_list[3]
                    pred = final_pred[-1].data.max(1)[1]
            else:
                cls_preds = net(blobs)


                if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                    pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

        test_predict_list += list(pred.cpu().numpy())
        test_target_list += list(targets)

    # 指标计算
    print('ACC:')
    print(metrics.accuracy_score(test_target_list, test_predict_list))
    print('SPE:')
    print(metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[0]))
    print('SEN:')
    print(metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[1]))


