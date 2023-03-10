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

'''
用于测试模型validation复现结果
发现问题在data set对于validation的sample是random的
'''
if __name__ == '__main__':
    train = False
    args = my_parse.parse_args()


    this_train_tag = datetime.now().strftime('%b%d_%H-%M-%S_') + args.net + '_' + args.modality + '_' + args.tag
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)
    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0


    # net = my_loadmodel.loadmodel(args)
    # net.froze_bn()

    # if args.weight is not None:
    # if args.tf_learning:
    print('Use transfer learning strategy!')
    # net.load_model(args.weight)
    # TODO: test
    checkpoint = torch.load('../Savings/save_models/Jan03_04-37-55_Ada_Ada_del_random_fix_initialized/best_val_acc.pth')
    # print("use pre-trained weight")
    net = checkpoint['net'].cuda()

    loss = my_loss.lossfun(args.loss)



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
        # if not args.random_fix:
        cudnn.benchmark = True


    # if args.random_fix:
    #     random_fix.seed_torch(seed=621)

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

                val_loss_strategy = loss(pred_list, cls_targets, args)

                final_pred = pred_list[1]

                pred = final_pred[-1].data.max(1)[1]

        test_predict_list += list(pred.cpu().numpy())
        test_target_list += list(targets)

        val_loss = val_loss_strategy
        val_loss_total = val_loss_total + val_loss

        utils.progress_bar(batch_idx, len(val_dataloader), 'Loss: %.5f | Acc: %.3f'
                           % (val_loss_total / (batch_idx + 1),
                              100. * metrics.accuracy_score(test_target_list, test_predict_list)))

    print(100. * metrics.accuracy_score(test_target_list, test_predict_list))
    print(val_loss_total)