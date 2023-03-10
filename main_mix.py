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
import numpy as np
'''
瞎猜acc=0.62
暂时将我们的模型命名为Ada
'''
if __name__ == '__main__':

    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    this_train_tag = datetime.now().strftime('%b%d_%H-%M-%S_') + args.net + '_' + args.modality + '_' + args.tag
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)
    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0


    net = my_loadmodel.loadmodel(args)
    # net.froze_bn()

    if args.weight is not None:
        if args.tf_learning:
            print('Use transfer learning strategy!')
            net.load_model(args.weight)
            # TODO: test
            # checkpoint = torch.load('../Savings/save_models/Dec16_05-48-34_Ada_Ada/best_val_acc.pth')
            # print("use pre-trained weight")
            # net = checkpoint['net'].cuda()

    loss = my_loss.lossfun(args.loss)


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
            targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            # blobs, cls_targets_0, cls_targets_1, lam = utils.mixup_data(blobs, targets)
            blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
            # y = torch.tensor(y)

            lam = np.random.beta(1.0, 1.0)

            index = torch.randperm(args.batch_size)
            blobs[0] = lam * blobs[0] + (1 - lam) * blobs[0][index, :]
            blobs[1] = lam * blobs[1] + (1 - lam) * blobs[1][index, :]
            # mixed_blobs = lam * blobs + (1 - lam) * blobs[index, :]
            cls_targets_0, cls_targets_1 = targets, targets[index]
            # return blobs, cls_targets_0, cls_targets_1, lam


            #
            # cls_targets_0 = utils.label_to_variable(cls_targets_0, gpu=GPU, volatile=False)
            # cls_targets_1 = utils.label_to_variable(cls_targets_1, gpu=GPU, volatile=False)



            # forward
            t0 = time.time()


            pred_list = net(blobs)
            # pred_list = hidden, cell, utility

            train_loss_strategy1 = loss(pred_list, cls_targets_0, args)
            train_loss_strategy2 = loss(pred_list, cls_targets_1, args)
            train_loss_strategy = lam * train_loss_strategy1 + (1-lam) * train_loss_strategy2


            optimizer.zero_grad()

            train_loss = train_loss_strategy
            train_loss_total = train_loss_total + train_loss
            train_loss.backward()
            optimizer.step()

            t1 = time.time()


            utils.progress_bar(batch_idx, len(train_dataloader), 'Epoch: %d | Lr: %.5f | Loss: %.5f '
                               % (epoch, lr, (train_loss_total / (batch_idx + 1))))

            # break
        '''
        直接在writer scalar中定义记录部分
        (self, loss, lr, data, n_iter, tag=None):
        '''
        tensorboard_writer.write_scalars(train_loss_total / len(train_dataloader),
                                         lr,
                                         data=None,
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
                            pred = final_pred[-1].data.max(1)[1]
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