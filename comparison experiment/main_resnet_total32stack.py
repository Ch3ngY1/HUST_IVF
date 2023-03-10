import os
import sys
sys.path.append("../")
import json
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
from torchvision.models import resnet50
'''
瞎猜acc=0.62
test --net Ada --tag resnet32totalstack -tf -rs --modality GRAY --num_segments 35 --lr 0.003 --epoch 75 -ld 25 --bs 3
'''

class MyResNet(nn.Module):
    def __init__(self, frames=1):
        super(MyResNet, self).__init__()
        # self.first = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.basemodel = resnet50(pretrained=True)
        # self.features = nn.Sequential(self.first, *list(self.basemodel.features.children())[1:])
        self.basemodel.conv1 = nn.Conv2d(frames, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.basemodel.fc = nn.Linear(512 * 4, 2)

    def forward(self, input):
        # x = self.first(input)
        x = self.basemodel(input)
        return x


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


    net = MyResNet(frames=args.num_segments)
    # net.froze_bn()

    # if args.weight is not None:
    #     if args.tf_learning:
    #         print('Use transfer learning strategy!')
    #         net.load_model(args.weight)
    #         # TODO: test
    #         # checkpoint = torch.load('../Savings/save_models/Dec16_05-48-34_Ada_Ada/best_val_acc.pth')
    #         # print("use pre-trained weight")
    #         # net = checkpoint['net'].cuda()

    loss = nn.CrossEntropyLoss()



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
            blobs = blobs.squeeze()
            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            # cls_targets = cls_targets.long()
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)

            # forward

            t0 = time.time()


            cls_preds = net(blobs)
            train_loss_strategy = loss(cls_preds, cls_targets.long())
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

            t1 = time.time()


            utils.progress_bar(batch_idx, len(train_dataloader), 'Lr: %.5f | Loss: %.3f | Acc: %.3f '
                               % (lr, train_loss_total / (batch_idx + 1), metrics.accuracy_score(train_target_list, train_predict_list)))
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
                blobs = blobs.squeeze()

                blobs = utils.img_data_to_variable(blobs, gpu=GPU)

                with torch.no_grad():

                    cls_preds = net(blobs)
                    val_loss_strategy = loss(cls_preds, cls_targets.long())
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