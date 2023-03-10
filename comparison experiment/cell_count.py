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
from torchvision.models import densenet201
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
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

    class mynet(nn.Module):
        def __init__(self):
            super(mynet, self).__init__()
            self.base = densenet201(pretrained=True)
            self.last = nn.Linear(1000,5)
            # self.base.classifier = nn.Linear(in_features=1920, out_features=1, bias=True)

        def forward(self, input):
            # input = input.view()
            output = self.base(input)
            output = self.last(output)
            return output
    net = mynet()
    # net = densenet201(pretrained=True)
    # net.classifier = nn.Linear(in_features=1920, out_features=5, bias=True)


    loss = FocalLoss(class_num=5)


    train_dataloader = my_dataloader.my_dataloader_stemcellcount(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem',
                                                       filename='train_.json', args=args, train=True)
    val_dataloader = my_dataloader.my_dataloader_stemcellcount(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem',
                                                 filename='valid_.json', args=args, train=False)
    # TODO: optimizer =
    optimizer = my_optimizer.init_optimizer(net=net, args=args)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Model\'s total number of parameters: %.3f M' % (num_params / 1e6))


    if GPU:
        net = net.cuda()

    best_val_acc = 0.0
    for epoch in range(args.epochs):

        train_predict_list, train_target_list = [], []
        net.train()
        train_loss_total = 0
        # Data
        for batch_idx, (blobs, targets) in enumerate(train_dataloader):
            # print(targets)
            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            cls_targets = cls_targets.long()
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)

            # forward
            t0 = time.time()

            # print(targets)
            cls_preds = net(blobs)
            train_loss_strategy = loss(cls_preds.squeeze(), cls_targets.squeeze())
            # TODO: mixup in loss function

            # if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
            #     pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
            # else:
            pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            train_predict_list += list(pred.cpu().numpy())
            train_target_list += list(targets)

            optimizer.zero_grad()

            train_loss = train_loss_strategy
            train_loss_total = train_loss_total + train_loss
            train_loss.backward()
            optimizer.step()

            t1 = time.time()


            utils.progress_bar(batch_idx, len(train_dataloader), 'Epoch: %d | Loss: %.5f | Acc: %.3f'
                               % (epoch, (train_loss_total / (batch_idx + 1)),
                                  100. * metrics.accuracy_score(train_target_list, train_predict_list)))

            # break
        '''
        直接在writer scalar中定义记录部分
        (self, loss, lr, data, n_iter, tag=None):
        '''
        tensorboard_writer.write_scalars(train_loss_total / len(train_dataloader),
                                         args.lr,
                                         [train_target_list, train_predict_list],
                                         n_iter=epoch,
                                         tag='train')

        savemodel.save_model(net, this_train_tag, 'last_checkpoint.pth', epoch, args.lr, False)
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
                                             args.lr,
                                             [test_target_list, test_predict_list],
                                             n_iter=epoch,
                                             tag='validation')

            if metrics.accuracy_score(test_target_list, test_predict_list) > best_val_acc:
                best_val_acc = metrics.accuracy_score(test_target_list, test_predict_list)
                print('Saving best model, acc:{}'.format(best_val_acc))
                savemodel.save_model(net, this_train_tag, 'best_val_acc.pth',
                                     epoch, args.lr, False, val_acc=best_val_acc)

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