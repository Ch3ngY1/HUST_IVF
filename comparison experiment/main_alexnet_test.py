import os
import sys
sys.path.append("../")
import json
import numpy as np
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel, random_fix
from utils.config import cfg
import time
from sklearn import metrics
from models import my_alexnet
import torch.nn as nn
'''
瞎猜acc=0.62
'''

def train():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    this_train_tag = 'alex' + args.tag
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)
    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0


    net = my_alexnet.Alex_()
    # net.froze_bn()

    # if args.weight is not None:
    #     if args.tf_learning:
    #         print('Use transfer learning strategy!')
    #         net.load_model(args.weight)
    #         # TODO: test
    #         # checkpoint = torch.load('../Savings/save_models/Dec16_05-48-34_Ada_Ada/best_val_acc.pth')
    #         # print("use pre-trained weight")
    #         # net = checkpoint['net'].cuda()

    loss = nn.MSELoss()

    if args.val_test:
        print('validation set is used for training for fitting test')
        train_dataloader = my_dataloader.my_dataloader_alex(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                       filename='valid.json', args=args, train=True)
    else:
        train_dataloader = my_dataloader.my_dataloader_alex(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                       filename='train.json', args=args, train=True)
    val_dataloader = my_dataloader.my_dataloader_alex(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                 filename='valid.json', args=args, train=False)
    # TODO: optimizer =
    optimizer = my_optimizer.init_optimizer(net=net, args=args)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Model\'s total number of parameters: %.3f M' % (num_params / 1e6))

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

        train_predict_list, train_target_list = [], []
        net.train()
        train_loss_total = 0
        # Data
        for batch_idx, (blobs, targets) in enumerate(train_dataloader):

            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            cls_targets = cls_targets.float()
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)

            # forward
            t0 = time.time()


            cls_preds = net(blobs)
            train_loss_strategy = loss(cls_preds.squeeze(), cls_targets.squeeze())
            # TODO: mixup in loss function

            # if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
            #     pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
            # else:
            #     pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            pred = (cls_preds>0.5).int()
            train_predict_list += list(pred.cpu().numpy())
            train_target_list += list(targets)

            optimizer.zero_grad()

            train_loss = train_loss_strategy
            train_loss_total = train_loss_total + train_loss
            train_loss.backward()
            optimizer.step()

            t1 = time.time()

            if not args.mixup:
                utils.progress_bar(batch_idx, len(train_dataloader), 'Epoch: %d | Lr: %.5f | Loss: %.5f | Acc: %.3f'
                                   % (epoch, lr, (train_loss_total / (batch_idx + 1)),
                                      100. * metrics.accuracy_score(train_target_list, train_predict_list)))
            else:
                utils.progress_bar(batch_idx, len(train_dataloader), 'Lr: %.5f | Loss: %.3f '
                                   % (lr, train_loss_total / (batch_idx + 1)))
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
        if not args.val_test:
            test_predict_list, test_target_list = [], []
            net.eval()
            val_loss_total = 0
            for batch_idx, (blobs, targets) in enumerate(val_dataloader):
                cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)


                blobs = utils.img_data_to_variable(blobs, gpu=GPU)

                with torch.no_grad():

                    cls_preds = net(blobs)
                    val_loss_strategy = loss(cls_preds.squeeze(), cls_targets.squeeze())
                    # val_loss_strategy = loss(cls_preds, cls_targets)
                    # TODO: mixup in loss function

                    if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                        pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                    else:
                        pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

                test_predict_list += list(pred.cpu().numpy())
                test_target_list += list(targets)

                val_loss = val_loss_strategy
                val_loss_total = val_loss_total + val_loss

                utils.progress_bar(batch_idx, len(val_dataloader), 'Loss: %.5f | Acc: %.3f | BestAcc: %.3f'
                                   % (val_loss_total / (batch_idx + 1),
                                      100. * metrics.accuracy_score(test_target_list, test_predict_list), best_val_acc))

            tensorboard_writer.write_scalars(train_loss_total / len(val_dataloader),
                                             lr,
                                             [test_target_list, test_predict_list],
                                             n_iter=epoch,
                                             tag='validation')

            if metrics.accuracy_score(test_target_list, test_predict_list) > best_val_acc:
                best_val_acc = metrics.accuracy_score(test_target_list, test_predict_list)
                print('Saving best model, acc:{}'.format(best_val_acc))
                savemodel.save_model(net, this_train_tag, 'best_val_acc.pth',
                                     epoch, lr, parallel_tag, val_acc=best_val_acc)

        #
        #     # save_model regularly
        #     if strategy_dict['regularly_save']:
        #         if step % (cfg.SOLVER.SAVE_INTERVAL * len(train_dataloader)) == 0 and step != start_step:
        #             print('Saving state regularly, iter:', step)
        #             save_model(net, resume_path, '{}_checkpoint.pth'.format(step), step, learning_rate, parallel_tag)
        # return valid_best_acc
        #
        #
        #
        #
    print('best validation acc = {}'.format(best_val_acc))
    # print('Ada\'s components are: cell_Trans={}, cell_Reward={}, cell_Pred={}'.format(args.cell_trans, args.cell_reward,
    #                                                                                   args.cell_pred))
    # default = ht cr cp
    print('Savings are saved at {}'.format(this_train_tag))


def val():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    GPU = 1
    net = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan12_05-41-01_Ada_Ada_useAlexNetasTest-notAda/best_val_acc.pth')['net']
    val_dataloader = my_dataloader.my_dataloader_alex(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                 filename='valid.json', args=args, train=False)

    net = net.cuda()
    best_val_acc = 0.0
    net.eval()

    # =========================================== validation ===========================================

    test_predict_list, test_target_list,val_predict_prob = [], [], []
    net.eval()
    for batch_idx, (blobs, targets) in enumerate(val_dataloader):
        cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)


        blobs = utils.img_data_to_variable(blobs, gpu=GPU)

        with torch.no_grad():

            cls_preds = net(blobs)

            if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
            else:
                pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability
        cls_preds = nn.functional.softmax(cls_preds, dim=1)
        test_predict_list += list(pred.cpu().numpy())
        test_target_list += list(targets)
        val_predict_prob += list(cls_preds.data.cpu().numpy())
        # val_loss = val_loss_strategy
        # val_loss_total = val_loss_total + val_loss

        utils.progress_bar(batch_idx, len(val_dataloader), 'Loss: %.5f | Acc: %.3f | BestAcc: %.3f'
                           % (0 / (batch_idx + 1),
                              100. * metrics.accuracy_score(test_target_list, test_predict_list), best_val_acc))
    print('acc={}'.format(metrics.accuracy_score(test_target_list, test_predict_list)))
    print('recall={}'.format([metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[cls])
            for cls in range(2)]))
    print('f1={}'.format(metrics.f1_score(test_target_list, test_predict_list,average='weighted')))
    # [metrics.f1_score(target_list, predict_list, average='weighted')]
    target_list_onehot = np.eye(2)[test_target_list]



    print('auc={}'.format(metrics.roc_auc_score(target_list_onehot, val_predict_prob)))
    #
    # target_list_onehot = np.eye(2)[test_target_list]
    # scalars += [metrics.roc_auc_score(target_list_onehot, pred_prob)]


    fpr, tpr, thersholds = metrics.roc_curve(test_target_list, [x[1] for x in val_predict_prob])
    print(fpr)
    print(tpr)
    print(metrics.auc(fpr, tpr))
if __name__ == '__main__':
    val()