from models import resnet
from models import densenet
from models import p3d_model
from models import i3dpt
import math
import torch.nn as nn
from models import Res_plus_p3d_plus_LSTM
#Res_plus_p3d_plus_GRU, Res_plus_p3d_plus_ViTCLS, \
    #Res_plus_p3d_plus_LSTM_duo,Res_plus_p3d_plus_LSTM_resgumbel, Dense_plus_p3d_plus_LSTM


def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        he(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        he(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def he(param):
    nn.init.kaiming_uniform(param)

def xavier(param):
    nn.init.xavier_uniform_(param)


def loadmodel(args, num_classes=2):
    if args.init_func == 'xavier':
        init_function = weights_init_xavier
    elif args.init_func == 'he':
        init_function = weights_init_he
    elif args.init_func == 'normal':
        init_function = weights_init_normal
    else:
        raise ValueError
    net_name = args.net
    if net_name == 'res50':
        net = resnet.resnet50(num_classes=num_classes, init_function=init_function)
    elif net_name == 'res101':
        net = resnet.resnet101(num_classes=num_classes, init_function=init_function)
    elif net_name == 'dense121':
        net = densenet.densenet121(num_classes=num_classes, init_function=init_function)
    elif net_name == 'dense169':
        net = densenet.densenet169(num_classes=num_classes, init_function=init_function)
    elif net_name == 'p3d199':
        net = p3d_model.P3D199(num_classes=num_classes, init_function=init_function,
                               action=args.action, modality=args.modality)
    elif net_name == 'i3d':
        net = i3dpt.i3d(num_classes=num_classes, init_function=init_function)
    elif net_name == 'Ada':
        if args.adamode == 'LSTM':
            net = Res_plus_p3d_plus_LSTM.Ada(num_classes=num_classes, init_function=init_function,
                                         num_frame=args.num_segments, c_trans=args.cell_trans, position_mode=args.position_add_mode, long=args.long, longweight=args.longweight)
        elif args.adamode == 'GRU':
            net = Res_plus_p3d_plus_GRU.Ada(num_classes=num_classes, init_function=init_function,
                                         num_frame=args.num_segments)
        elif args.adamode == 'transformer':
            net = Res_plus_p3d_plus_ViTCLS.Ada_LSTM(num_classes=num_classes, init_function=init_function,
                                         num_frame=args.num_segments)
        elif args.adamode == 'bi':
            net = Res_plus_p3d_plus_LSTM_duo.Ada(num_classes=num_classes, init_function=init_function,
                                         num_frame=args.num_segments)
        elif args.adamode == 'res':
            net = Res_plus_p3d_plus_LSTM_resgumbel.Ada(num_classes=num_classes, init_function=init_function,
                                                 num_frame=args.num_segments)
        elif args.adamode == 'dense':
            net = Dense_plus_p3d_plus_LSTM.Ada(num_classes=num_classes, init_function=init_function,
                                                 num_frame=args.num_segments)
    elif net_name == 'AdaViT':

        net = Res_plus_p3d_plus_ViTCLS.Ada(num_classes=num_classes, init_function=init_function,
                                         num_frame=args.num_segments)

    else:
        raise KeyError('Unknown model: {}'.format(net_name))
    return net


if __name__ == '__main__':
    import torch
    from models import Res_plus_p3d_plus_LSTM
    input_img = torch.rand([3, 128, 1, 224, 224])
    input_pos = torch.rand([3, 128, 2048])
    input_ = [input_img, input_pos]
    net = Res_plus_p3d_plus_LSTM.Ada(init_function=weights_init_he)
    net.load_model('/data2/chengyi/.torch/models/resnet50-19c8e357.pth')
    out = net(input_)

