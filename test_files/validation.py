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
from sklearn import metrics
# from torch import torch.loa

'''
瞎猜acc=0.62
暂时将我们的模型命名为Ada
'''
if __name__ == '__main__':

    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=18518)



    this_train_tag = datetime.now().strftime('%b%d_%H-%M-%S_') + args.net + '_' + args.modality + '_' + args.tag
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)
    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0


    net = my_loadmodel.loadmodel(args)
    # net.froze_bn()

    net.load_model(args.weight)
    # TODO: test
    checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan03_04-37-55_Ada_Ada_del_random_fix_initialized/best_val_acc.pth')

    net = checkpoint['net'].cuda()

    loss = my_loss.lossfun(args.loss)


    val_dataloader = my_dataloader.my_dataloader(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                 filename='valid.json', args=args, train=False)
    # TODO: optimizer =
    optimizer = my_optimizer.init_optimizer(net=net, args=args)
    num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print('Model\'s total number of parameters: %.3f M' % (num_params / 1e6))

    parallel_tag = False
    if GPU:
        net = net.cuda()
        if args.gpus is not None:
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
            if args.modality == 'Ada':
                assert isinstance(args.gpus, list) and len(args.gpus) > 1
                net = torch.nn.DataParallel(net, device_ids=list(range(len(args.gpus))))
            else:
                assert isinstance(args.gpus, list) and len(args.gpus) > 1
                net.parallel(args.gpus)
            parallel_tag = True
        # if not args.random_fix:
        #     cudnn.benchmark = True

        # Data
    import numpy as np
    offset_list = np.arange(-0.06,0.06,0.005)
    for result_offset in offset_list:
    # result_offset = 0.1
        print('Pred Offset = {}'.format(result_offset))
        predict_offset = torch.tensor([result_offset, 0]).cuda()
        predict_offset.repeat(args.batch_size,1)

        test_predict_list, test_target_list = [], []
        net.eval()
        val_loss_total = 0
        for batch_idx, (blobs, targets) in enumerate(val_dataloader):
            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            if args.modality == 'Ada':
                blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
            else:
                blobs = utils.img_data_to_variable(blobs, gpu=GPU)

            with torch.no_grad():
                if args.net == 'Ada':
                    pred_list = net(blobs)
                    if not args.adamode == 'bi':
                        val_loss_strategy = loss(pred_list, cls_targets, args)
                    else:
                        loss1 = loss(pred_list[0:3], cls_targets, args)
                        loss2 = loss(pred_list[3:], cls_targets, args)
                        val_loss_strategy = loss1 + loss2
                    if args.cell_pred:
                        final_pred = pred_list[1]
                        if args.adamode == 'bi':
                            final_pred = pred_list[1] + pred_list[4]
                        # pred = final_pred[-1].data.max(1)[1]
                        pred_origin = final_pred[-1].data
                        pred_origin += predict_offset
                        pred = pred_origin.max(1)[1]
                    else:
                        final_pred = pred_list[0]
                        if args.adamode == 'bi':
                            final_pred = pred_list[0] + pred_list[3]
                        pred = final_pred[-1].data.max(1)[1]
                else:
                    cls_preds = net(blobs)
                    val_loss_strategy = loss(cls_preds, cls_targets)
                    # TODO: mixup in loss function

                    if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                        pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                    else:
                        pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            test_predict_list += list(pred.cpu().numpy())
            test_target_list += list(targets)

            # val_loss = val_loss_strategy
            # val_loss_total = val_loss_total + val_loss

            utils.progress_bar(batch_idx, len(val_dataloader), 'Acc: %.3f'
                               % (100. * metrics.accuracy_score(test_target_list, test_predict_list)))

        print(metrics.accuracy_score(test_target_list, test_predict_list))

        print(metrics.recall_score(test_target_list, test_predict_list, average='macro',labels=[0]))

        print(metrics.recall_score(test_target_list, test_predict_list, average='macro',labels=[1]))


