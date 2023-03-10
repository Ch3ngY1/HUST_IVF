from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) * 0.25
            self.alpha[0] = 1 - 0.25
            self.alpha = Variable(self.alpha)
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
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def lossfun(args):
    lossname = args.loss
    lossname = lossname.lower()
    if lossname == 'cross_entropy':
        lossfunction = CrossEntropyLoss()
    elif lossname == 'focal_loss':
        lossfunction = FocalLoss()
    elif lossname == 'ada_loss':
        if args.utility_weight == 0 and args.reward_weight == 0:
            print('Loss Function = Loss_pred(CrossEntropy)')
        elif args.utility_weight == 0:
            print('Loss Function = Loss_pred(CrossEntropy) - Loss_Reward(Accumulate)')
        elif args.reward_weight == 0:
            print('Loss Function = Loss_pred(CrossEntropy) + Loss_Utility(L2)')
        else:
            print('Loss Function = Loss_pred(CrossEntropy) + Loss_Utility(L2) - Loss_Reward(Accumulate)')
        # lossfunction = [CrossEntropyLoss(), cal_reward, ada_utility, reward_loss]
        lossfunction = AdaLoss
    else:
        raise KeyError('Unknown loss: {}'.format(lossname))
    return lossfunction


def AdaLoss(output, cls_targets, args):
    # loss(cls_preds, cls_targets)

    if args.loss_reward_entropy:
        lossfunction = [CrossEntropyLoss(), cal_reward_entropy, ada_utility, reward_loss]
    else:
        # TODO: Try focal loss
        # lossfunction = [FocalLoss(), cal_reward, ada_utility, reward_loss]
        lossfunction = [CrossEntropyLoss(), cal_reward, ada_utility, reward_loss]
    # pred_raw = hidden
    if args.adamode == 'GRU':
        hidden, utility = output
    else:
        hidden, cell, utility, watch = output

    if args.cell_hidden_ratio:
        print('======mix======')
        ratio = args.cell_hidden_ratio
        loss_pred = lossfunction[0](cell[-1] * ratio + hidden[-1] * (1 - ratio), cls_targets)
        cellhidden = [c * ratio + h * (1 - ratio) for c, h in zip(cell, hidden)]
        reward = lossfunction[1](cellhidden, cls_targets)
    else:
        # TODO: 可以直接把cp/cr融合到ratio=1/0
        if args.cell_pred:
            loss_pred = lossfunction[0](cell[-1], cls_targets)
        else:
            loss_pred = lossfunction[0](hidden[-1], cls_targets)

        if args.cell_reward:
            reward = lossfunction[1](cell, cls_targets, watch)
        else:
            reward = lossfunction[1](hidden, cls_targets, watch)
    loss_utility = lossfunction[2](utility, reward)
    loss_reward = lossfunction[3](reward)

    # TODO： 消融实验1
    # return loss_pred
    # TODO： 消融实验3:Loss=loss+ reward
    # return_loss = loss_pred - args.reward_weight * loss_reward
    # return_loss = loss_pred + args.utility_weight * loss_utility - args.reward_weight * loss_reward  # 72.42% weight=1

    # 感觉就算把weight设置为0 但是也会涉及到部分反向传播，因此尝试if 0 --> 直接不加入这部分loss
    if args.reward_weight == 0:
        loss_reward = 0
    if args.utility_weight == 0:
        loss_utility = 0
    return_loss = loss_pred + args.utility_weight * loss_utility - args.reward_weight * loss_reward  # 72.42% weight=1


    # # return_loss = loss_pred + loss_utility # TODO:at Adaco
    # return_loss = loss_pred - args.reward_weight * loss_reward  # 71.21% TODO:at Ada Retest
    # # return_loss = loss_reward  # --> does not have gradient
    # # return_loss = loss_pred # 64.24%

    return return_loss


def cal_reward(pred, label, watch):
    # pred = list: F * tensor(N, 2)
    # label = N * tensor(1)
    relu = nn.ReLU(inplace=False)
    pred = torch.stack(pred, dim=0)  # --> tensor (F, N, 2)
    pred = pred.transpose(1, 0)  # --> tensor (N, F, 2)
    bs, frame, _ = pred.shape
    pred = F.softmax(pred, dim=2)
    # watch = 1 --> use
    watch = torch.stack(watch).permute(1, 0, 2).cuda()
    watch = torch.cat([torch.ones(bs, 1).cuda(), watch[:, :, 1]], dim=1)

    mt_list = []
    for i in range(bs):
        # prob difference
        prob_difference = pred[i, :, label[i]] - pred[i, :, 1 - label[i]]
        # prob_difference = torch.mul(prob_difference, watch[i])
        mt_list.append(prob_difference)
    mt = torch.stack(mt_list, dim=0)  # --> N * F

    reward_all = []
    for j in range(bs):
        reward = []
        for i in range(frame):
            if i == 0:
                reward.append(relu(mt[j, i]))
            else:
                # 当前的mt-之前mt的最大值， max（value， 0）
                current_max = max(mt[j, 0:i])
                reward.append(relu(mt[j, i] - current_max))
        reward = torch.stack(reward) # 2022-2-15 # 22-2-22
        reward_all.append(reward)
    # return torch.tensor(reward_all) # 2022-2-15 # 22-2-22
    return torch.stack(reward_all) # 22-2-22


def cal_reward_entropy(pred, label):
    # pred = list: F * tensor(N, 2)
    # label = N * tensor(1)
    relu = nn.ReLU(inplace=False)
    pred = torch.stack(pred, dim=0)  # --> tensor (F, N, 2)
    pred = pred.transpose(1, 0)  # --> tensor (N, F, 2)
    bs, frame, _ = pred.shape
    pred = F.softmax(pred, dim=2)

    mt = -pred[:, :, 0] * torch.log(pred[:, :, 0]) - pred[:, :, 1] * torch.log(pred[:, :, 1])
    mt = -mt
    # --> N * F
    reward_all = []
    for j in range(bs):
        reward = []
        for i in range(frame):
            if i == 0:
                # reward.append(mt[j,i])
                reward.append(relu(mt[j, i]))
            else:
                current_max = max(mt[j, 0:i])
                reward.append(relu(mt[j, i] - current_max))
        reward_all.append(reward)

    return



def reward_loss(reward):
    # reward = N * F
    # return reward.mean(dim=1).sum()
    return reward.sum()


def ada_utility(utility_pred, reward):
    bs, frame = reward.shape
    utility_all = []
    for j in range(bs):
        utility = []
        for i in range(frame):
            exp = torch.arange(0., float(frame - i))
            base = torch.tensor(0.9)
            u = (reward[j, i:].cpu() * base.pow(exp)).mean()
            utility.append(u)
        utility_all.append(utility)
    utility_all = torch.tensor(utility_all).cuda()
    utility_pred = torch.stack(utility_pred).squeeze(dim=-1).transpose(1, 0)  # --> N * F
    loss_sum = ((utility_all - utility_pred) * (utility_all - utility_pred)).sum() / 2
    loss_fn = nn.MSELoss()
    loss_mean = loss_fn(utility_all, utility_pred)
    return loss_sum  # loss_mean
