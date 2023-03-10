import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel
from utils.config import cfg
import time
import re
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
import argparse
'''
瞎猜acc=0.62
暂时将我们的模型命名为Ada
'''
if __name__ == '__main__':

    args = my_parse.parse_args()
    this_train_tag = datetime.now().strftime('%b%d_%H-%M-%S_') + args.net + '_' + args.modality
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)
    GPU = True
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # 1. 获取环境信息
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    node_list = str(os.environ['SLURM_NODELIST'])

    # 对ip进行操作
    node_parts = re.findall('[0-9]+', node_list)
    host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])

    # 注意端口一定要没有被使用
    port = "23456"

    # 使用TCP初始化方法
    init_method = 'tcp://{}:{}'.format(host_ip, port)

    # 多进程初始化,初始化通信环境
    dist.init_process_group("nccl", init_method=init_method,
                            world_size=world_size, rank=rank)

    # 指定每个节点上的device
    torch.cuda.set_device(local_rank)

    net = my_loadmodel.loadmodel(args)


        # to store data and weight in cuda: args.gpus[0] instead of cuda:0


    if args.weight is not None:
        if args.tf_learning:
            print('Use transfer learning strategy!')
            net.load_model(args.weight)

    loss = my_loss.lossfun(args.loss)

    if args.val_test:
        print('validation set is used for training for fitting test')
        train_dataloader = my_dataloader.my_dataloader(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                       filename='valid.json', args=args, train=True)
    else:
        train_dataloader = my_dataloader.my_dataloader(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                       filename='train.json', args=args, train=True)
    val_dataloader = my_dataloader.my_dataloader(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
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
                net = DDP(net, device_ids=[local_rank])
            else:
                assert isinstance(args.gpus, list) and len(args.gpus) > 1
                net.parallel(args.gpus)
            parallel_tag = True
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

            if args.mixup:
                blobs, cls_targets_0, cls_targets_1, lam = utils.mixup_data(blobs, targets)
                cls_targets_0 = utils.label_to_variable(cls_targets_0, gpu=GPU, volatile=False)
                cls_targets_1 = utils.label_to_variable(cls_targets_1, gpu=GPU, volatile=False)
                cls_targets = [cls_targets_0, cls_targets_1, lam]
            else:
                cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            if args.modality == 'Ada':
                blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
            else:
                image_data = utils.img_data_to_variable(blobs, gpu=GPU)

            # forward
            t0 = time.time()

            if args.modality == 'Ada':
                pred_list = net(blobs)
                train_loss_strategy = loss(pred_list, cls_targets)

                pred = pred_list[0][-1].data.max(1)[1]
            else:
                cls_preds = net(image_data)
                train_loss_strategy = loss(cls_preds, cls_targets)
                # TODO: mixup in loss function

                if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                    pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            train_predict_list += list(pred.cpu().numpy())
            train_target_list += list(targets)

            optimizer.zero_grad()
            if args.modality == 'AdapPool':
                loss_ratio = 0.2
                loss_total = train_loss_strategy['cls_loss'] + loss_ratio * train_loss_strategy['gamma_loss']
                loss_total.backward()
            else:
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
                if args.modality == 'Ada':
                    blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
                else:
                    image_data = utils.img_data_to_variable(blobs, gpu=GPU)

                with torch.no_grad():
                    if args.modality == 'Ada':

                        pred_list = net(blobs)
                        val_loss_strategy = loss(pred_list, cls_targets)

                        pred = pred_list[0][-1].data.max(1)[1]
                    else:
                        cls_preds = net(image_data)
                        val_loss_strategy = loss(cls_preds, cls_targets)
                        # TODO: mixup in loss function

                        if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                            pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                        else:
                            pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

                test_predict_list += list(pred.cpu().numpy())
                test_target_list += list(targets)

                val_loss = val_loss_strategy
                val_loss_total = val_loss_total + val_loss

                utils.progress_bar(batch_idx, len(val_dataloader), 'Loss: %.5f | Acc: %.3f'
                                   % (val_loss_total / (batch_idx + 1),
                                      100. * metrics.accuracy_score(test_target_list, test_predict_list)))

            tensorboard_writer.write_scalars(train_loss_total / len(val_dataloader),
                                             lr,
                                             [test_target_list, test_predict_list],
                                             n_iter=epoch,
                                             tag='validation')

            if metrics.accuracy_score(test_target_list, test_predict_list) > best_val_acc:
                best_val_acc = metrics.accuracy_score(test_target_list, test_predict_list)
                print('Saving best model, acc:{}'.format(best_val_acc))
                savemodel.save_model(net, this_train_tag, 'best_val_acc.pth', epoch, lr, parallel_tag)



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
