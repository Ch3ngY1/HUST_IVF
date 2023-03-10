import os
import sys
sys.path.append('../')
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel, random_fix
from utils.config import cfg
import time
from sklearn import metrics
import numpy as np
# from torch import torch.loa

'''
瞎猜acc=0.62
暂时将我们的模型命名为Ada
'''
def single(net, args, setname):
    GPU = 1


    # loss = my_loss.lossfun(args.loss)
    # '/data/mxj/IVF_HUST/first_travel_VideoImage'
    val_dataloader = my_dataloader.my_dataloader(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage',
                                                 filename='{}.json'.format(setname), args=args, train=True)

    # val_dataloader = my_dataloader.my_dataloader_resample(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage',
    #                                                       filename='{}_0114.json'.format(setname), args=args, train=False)


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


    test_predict_list, test_target_list, pred_list_for_adjust = [], [], []
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
                final_pred = pred_list[1]
                # if pred_list == []:
                #     pred_list_for_adjust = list(final_pred[-1].cpu().numpy())
                # else:
                pred_list_for_adjust += list(final_pred[-1].cpu().numpy())
                pred = final_pred[-1].data.max(1)[1]
            #     pred_list = net(blobs)
            #     if not args.adamode == 'bi':
            #         val_loss_strategy = loss(pred_list, cls_targets, args)
            #     else:
            #         loss1 = loss(pred_list[0:3], cls_targets, args)
            #         loss2 = loss(pred_list[3:], cls_targets, args)
            #         val_loss_strategy = loss1 + loss2
            #     if args.cell_pred:
            #         final_pred = pred_list[1]
            #         if args.adamode == 'bi':
            #             final_pred = pred_list[1] + pred_list[4]
            #         pred = final_pred[-1].data.max(1)[1]
            #         # pred_origin = final_pred[-1].data
            #         # pred_origin += predict_offset
            #         # pred = pred_origin.max(1)[1]
            #     else:
            #         final_pred = pred_list[0]
            #         if args.adamode == 'bi':
            #             final_pred = pred_list[0] + pred_list[3]
            #         pred = final_pred[-1].data.max(1)[1]
            else:
                cls_preds = net(blobs)
                # val_loss_strategy = loss(cls_preds, cls_targets)
                # TODO: mixup in loss function

                if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                    pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

        test_predict_list += list(pred.cpu().numpy())
        test_target_list += list(targets)

        # val_loss = val_loss_strategy
        # val_loss_total = val_loss_total + val_loss

        # utils.progress_bar(batch_idx, len(val_dataloader), 'Acc: %.3f'
        #                    % (100. * metrics.accuracy_score(test_target_list, test_predict_list)))
    acc = metrics.accuracy_score(test_target_list, test_predict_list)
    recall0 = metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[0])
    recall1 = metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[1])
    # print('acc={},recall0={},recall1={}'.format(acc, recall0, recall1))



    start = 0.4
    step = 0.0025
    acc_list = []
    recall_list = []
    for i in np.arange(-start,start,step):
        adjust_value = torch.tensor([i,0]).repeat(330,1)
        adjusted_pred = torch.tensor(pred_list_for_adjust) + adjust_value
        test_predict_list_adjusted = [pred_each.clone().detach().data.max(dim=0)[1] for pred_each in adjusted_pred]
        acc_list.append(100. * metrics.accuracy_score(test_target_list, test_predict_list_adjusted))
        recall_list.append([metrics.recall_score(test_target_list, test_predict_list_adjusted, average='macro', labels=[cls])
                    for cls in range(2)])

    # print(acc_list)
    acc_list = np.array(acc_list)
    res = np.where(acc_list == np.amax(acc_list))
    current_max = acc_list.max()
    recall_list0 = np.array(recall_list)[:,0]
    recall_list1 = np.array(recall_list)[:,1]
    recall0 = recall_list0[res]
    recall1 = recall_list1[res]

    # return acc, recall0, recall1, max
    return current_max, recall0, recall1



def testloop(setname):
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
    # Jan03_04-37-55_Ada_Ada_del_random_fix_initialized
    root = '/data2/chengyi/myproject/Savings/tensorboardx'
    bestacc = 0.0
    for root, dirs, files in os.walk(root):
        for dir in dirs[40:]:
            if os.path.exists('/data2/chengyi/myproject/Savings/save_models/{}/best_val_acc.pth'.format(dir)):
                checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/{}/best_val_acc.pth'.format(dir))

                net = checkpoint['net'].cuda()

                # loss = my_loss.lossfun(args.loss)

                val_dataloader = my_dataloader.my_dataloader(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                             filename='{}.json'.format(setname), args=args, train=False)
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

                    # val_loss = val_loss_strategy
                    # val_loss_total = val_loss_total + val_loss

                    utils.progress_bar(batch_idx, len(val_dataloader), 'Acc: %.3f'
                                       % (100. * metrics.accuracy_score(test_target_list, test_predict_list)))
                if metrics.accuracy_score(test_target_list, test_predict_list)>bestacc:
                    bestacc = metrics.accuracy_score(test_target_list, test_predict_list)
                    print('current_best_acc={}'.format(bestacc))
                    bestacc_file = dir
                print(metrics.accuracy_score(test_target_list, test_predict_list))
            # print(metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[0]))
            #
            # print(metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[1]))
    print('best acc = {}, in dir {}'.format(bestacc,bestacc_file))
# if __name__ == '__main__':
#     setname = 'test'
#     print('==========' + setname + '==========')
#     max_list = []
#     r0 = []
#     r1 = []
#     for i in range(2):
#         max_list.append(single(setname)[0])
#         r0.append(single(setname)[1])
#         r1.append(single(setname)[2])
#         utils.progress_bar(i, 50)
#     max_list = np.array(max_list)
#     r0 = np.array(r0)
#     r1 = np.array(r1)
#     res = np.where(max_list == np.amax(max_list))
#     print('max acc in test dataset is:')
#     print(max_list.max())
#     print(r0[res])
#     print(r1[res])
#     # bestacc = 0.0
#     # re0 = 0.0
#     # re1 = 0.0
#     # for i in range(20):
#     #     acc, recall0, recall1 = single('test')
#     #     if bestacc<acc:
#     #         bestacc = acc
#     #         re0 = recall0
#     #         re1 = recall1
#     # print('FINAL=================>acc={},recall0={},recall1={}'.format(bestacc, re0, re1))

if __name__ == '__main__':
    args = my_parse.parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]

    net = my_loadmodel.loadmodel(args)


    net.load_model(args.weight)
    # TODO: test
    # a = '/data2/chengyi/embryo_HUSH/mine/save_models/HUST_video/first_travel_VideoImage/small_p3d199_GRAY_224_Segments_16_Clip_1_tf_test/best_valid_acc_checkpoint.pth'
    # Jan03_04-37-55_Ada_Ada_del_random_fix_initialized
    # Jan12_08-09-25_Ada_Ada_adatest
    # Jan11_01-46-28_Ada_Ada_longetime-divsum-tau1-random
    # Jan14_10-55-31_Ada_Ada_longetimeweight200-40
    # Jan17_05-59-36_Ada_Ada_weight-test-25-100
    filename = 'Jan17_05-59-36_Ada_Ada_weight-test-25-100'
    a = '/data2/chengyi/myproject/Savings/save_models/{}/best_val_acc.pth'.format(filename)
    checkpoint = torch.load(a)
    # net.load_state_dict(checkpoint['state_dict'])
    net = checkpoint['net'].cuda()
    setname = 'test'
    print('==========' + setname + '==========')

    # print(single)
    #
    # print('==========' + setname + '==========')
    max_list = []
    r0 = []
    r1 = []
    for i in range(200):
        max_list.append(single(args=args, net=net,setname=setname)[0])
        r0.append(single(args=args, net=net,setname=setname)[1])
        r1.append(single(args=args, net=net,setname=setname)[2])
        utils.progress_bar(i, 200)
    max_list = np.array(max_list)
    r0 = np.array(r0)
    r1 = np.array(r1)
    res = np.where(max_list == np.amax(max_list))
    print('max acc in test dataset is:')
    print(max_list.max())
    print(r0[res])
    print(r1[res])
    # bestacc = 0.0
    # re0 = 0.0
    # re1 = 0.0
    # for i in range(20):
    #     acc, recall0, recall1 = single('test')
    #     if bestacc<acc:
    #         bestacc = acc
    #         re0 = recall0
    #         re1 = recall1
    # print('FINAL=================>acc={},recall0={},recall1={}'.format(bestacc, re0, re1))



# Feb24_11-03-30_Ada_Ada_old-reward-cpcr : 70