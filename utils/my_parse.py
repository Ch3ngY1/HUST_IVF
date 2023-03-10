import argparse
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a network')
    # model architecture('attention' -> 'cross_attention')
    parser.add_argument('arc', choices=['single', 'fastlrdecay', 'small', 'Video', 'AdapPool', 'slowfast', 'test'],
                        help='the architecture of model')
    parser.add_argument('--net', dest='net',
                        choices=['vgg16bn', 'res50', 'res101', 'dense121', 'dense169', 'p3d199', 'i3d', 'myres50',
                                 'myres101', 'AdapPool', 'slowfast', 'Ada', 'AdaViT'],
                        default='res50', type=str)
    parser.add_argument('--init_func', dest='init_func', default='xavier',
                        choices=['xavier', 'he', 'normal'], type=str,
                        help='initialization function of model')
    # data parameters
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--num_segments', dest='num_segments',
                        help='tag of the model',
                        default=32, type=int)
    parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
    parser.add_argument('--flow_prefix', type=str, help="prefix of Flow frames", default='flow_')
    parser.add_argument('--modality', type=str, help='input modality of networks', default='RGB',
                        choices=['RGB', 'Flow', 'RGBDiff', 'GRAY', 'GrayAnother', 'CANCELLED', 'SPECTRUM',
                                 'FlowPlusGray', 'VideoGray', 'AdapPool', 'slowfast', 'CSP', 'LSTM','Ada'])
    # CST: continuously sampling at each sampled period
    parser.add_argument('-rd', '--random_sample', dest='random_sample', action='store_true',
                        help='whether to use random sampling strategy during each segment')
    parser.add_argument('-vt', '--val_test', dest='val_test', action='store_true',
                        help='use validation set to train model')
    # training strategy
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pre-trained model weights',
                        default=None,
                        type=str)
    parser.add_argument('-tf', '--tf_learning', dest='tf_learning', action='store_true',
                        help='whether to use transfer learning')
    parser.add_argument('--resume', dest='resume',
                        help='resume checkpoint',
                        default=None, type=int)
    parser.add_argument('--auto', dest='auto', action='store_true',
                        help='whether to use auto learning decay')
    parser.add_argument('--auto_metric', dest='auto_metric',
                        help='when use "auto", select a metric as every epoch s judgement',
                        default='none', choices=['none', 'loss', 'acc', 'weighted_f1_score'])
    parser.add_argument('-rs', '--regularly_save', dest='regularly_save', action='store_true',
                        help='whether to save checkpoint timely')
    parser.add_argument('--kfold', dest='kfold', default=None, type=int,
                        help='Use k-fold cross-validation')
    parser.add_argument('--lr', dest='lr', default=0.0003, type=float, help='learning rate of training')
    parser.add_argument('--loss', dest='loss', default='Cross_Entropy', type=str,
                        choices=['Cross_Entropy', 'Focal_Loss', 'Balance_Cross_Entropy', 'ada_loss'],
                        help='loss type used in training stage')
    parser.add_argument('--class_weights', dest='class_weights', type=int, nargs='+', default=None,
                        help='Loss weights which are used to optimize the less class')
    parser.add_argument('-m', '--merge_valid', dest='merge_valid', action='store_true',
                        help='whether to merge valid and test set during evluation')
    parser.add_argument('-dl', '--dynamic_lr', dest='dynamic_lr', action='store_true',
                        help='whether to use dynamic lr during training(p3d)')
    parser.add_argument('--ld', dest='learning_rate_decay', type=int, default=None,
                        help='every ld epochs, lr = lr * 0.3')
    # gpu setting
    parser.add_argument('--gpus', dest='gpus', default=None, nargs='+', type=int,
                        help='When use dataparallel, it is needed to set gpu Ids')
    parser.add_argument('--debug', action='store_true',
                        help='to prevent record acc in record.txt')
    parser.add_argument('--mixup', action='store_true',
                        help='to use mixup data process-method')
    parser.add_argument('--action', action='store_true',
                        help='to use action module')
    parser.add_argument('--mixtureimg', default=None, type=str, choices=['single', 'couple'],
                        help='to use mixture on image input')
    parser.add_argument('--bs', dest='batch_size', default=3, type=int)
    parser.add_argument('--mt', dest='momentum', default=0.9, type=float)
    parser.add_argument('--wd', dest='weight_decay', default=0.0001, type=float)
    parser.add_argument('-cp', dest='cell_pred', action='store_true')
    parser.add_argument('--chr', dest='cell_hidden_ratio', default=None, type=float)
    parser.add_argument('-cr', dest='cell_reward', action='store_true')
    parser.add_argument('-ct', dest='cell_trans', action='store_true')
    parser.add_argument('--epoch', dest='epochs', default=75, type=int)
    parser.add_argument('-rf', dest='random_fix', action='store_true')
    parser.add_argument('--pool_factor', default='1111', type=str, choices=['0000', '1000', '1100', '1110', '1111'],
                        help='to use 1=max, 0=min pool_method')
    parser.add_argument('--adamode', default='LSTM', type=str, choices=['LSTM', 'GRU', 'transformer', 'bi', 'res','dense'])
    # ================== POSITIONAL ENCODING ==================
    parser.add_argument('--pm', dest='positionmode', default=None, choices=['850', '32', 'embedding'], type=str)
    parser.add_argument('--pam', dest='position_add_mode', default=None, choices=['add', 'cat'], type=str)
    # ================== LOSS ==================
    parser.add_argument('--rw', dest='reward_weight', default=1.0, type=float)
    parser.add_argument('--uw', dest='utility_weight', default=1.0, type=float)
    parser.add_argument('-lre', dest='loss_reward_entropy', action='store_true')
    # ================== LONG TIME ==================
    parser.add_argument('-long', dest='long', action='store_true')
    parser.add_argument('-longw', dest='longweight', action='store_true')

    # ================================================ DDP ================================================
    parser.add_argument("--local_rank", default=-1)
    # ================================================ DDP ================================================

    # if len(sys.argv) == 1:
    #   parser.print_help()
    #   sys.exit(1)
    args = parser.parse_args()
    weight_choose(args)
    return args

def weight_choose(args):
    if args.net == 'res50':
        args.weight = '/data2/chengyi/.torch/models/resnet50-19c8e357.pth'
    elif args.net == 'res101':
        args.weight = '/home/mxj/.torch/models/resnet101-5d3b4d8f.pth'
    elif args.net in ['myres50', 'myres101']:
        args.weight = None
    elif args.net == 'AdapPool':
        args.weight = None
    elif args.net == 'vgg16bn':
        args.weight = '/home/mxj/.torch/models/vgg16-397923af.pth'
    elif args.net == 'dense121':
        args.weight = '/home/mxj/.torch/models/densenet121-241335ed.pth'
    elif args.net == 'dense169':
        args.weight = '/home/mxj/.torch/models/densenet169-6f0f7f60.pth'
    elif args.net == 'res50_aargs.weightme':
        args.weight = '/home/mxj/.torch/models/resnet50-19c8e357.pth'
    elif args.net == 'p3d199':
        if args.modality in ['RGB', 'CANCELLED', 'GRAY', 'GrayAnother']:
            # Kinetics600
            # args.weight = '/home/mxj/data/CommonWeights/p3d_rgb_199.checkpoint.pth.tar'
            # args.weight = '/home/chengyi/fsdownload/tmp_video_classification/save_models/HUST_video/first_travel_VideoImage/single_p3d199_CANCELLED_224_Segments_16_Clip_1_tf_test/3300_checkpoint.pth'
            args.weight = '/data/mxj/CommonWeights/P3D199_rgb_299x299_model_best.pth.tar'
        elif args.modality == 'SPECTRUM':
            args.weight = None
        else:
            args.weight = '/data/mxj/CommonWeights/p3d_flow_199.checkpoint.pth.tar'

    elif args.net == 'i3d':
        if args.modality in ['RGB', 'SPECTRUM', 'CANCELLED', 'GRAY', 'GrayAnother']:
            args.weight = '/data/grq/project/I3D/model/model_rgb.pth'
        else:
            args.weight = '/data/grq/project/I3D/model/model_flow.pth'
    elif args.net == 'slowfast':
        args.weight = None
    elif args.net in ['Ada', 'AdaViT']:
        args.weight = '/data2/chengyi/.torch/models/resnet50-19c8e357.pth'
    elif args.adamode == 'dense':
        args.weight = '/home/chengyi/.cache/torch/checkpoints/densenet201-c1103571.pth '
    else:
        print('unexpected BaseModel!')
        raise KeyError