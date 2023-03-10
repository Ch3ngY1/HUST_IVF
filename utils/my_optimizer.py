import torch
from models.p3d_model import get_optim_policies

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_optimizer(net, args):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    parameters = []
    for para in net.parameters():
        if para.requires_grad == True:
            parameters.append(para)
    if args.net == 'p3d199' and args.dynamic_lr:
        optimizer = torch.optim.SGD(get_optim_policies(net.model, args.modality), momentum=args.momentum,
                                    lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(parameters, momentum=args.momentum, lr=args.lr,
                                    weight_decay=args.weight_decay)
    return optimizer