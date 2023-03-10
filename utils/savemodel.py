from utils.config import cfg
import os
import torch


def save_model(net, path_name, model_name, epoch, lr, parallel=False, val_acc=None):
    save_dict = {
        'net': net.module if parallel else net,
        'state_dict': net.module.state_dict() if parallel else net.state_dict(),
        'lr': lr,
        'epoch': epoch,
        'val_acc': val_acc,
    }

    root_path = cfg.ModelSave_Path
    load_path = os.path.join(root_path, path_name)
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    load_path = os.path.join(load_path, model_name)

    # if parallel:
    #     save_dict = {
    #         'iters': epoch,
    #         'state_dict': net.module.state_dict(),
    #         'lr': lr
    #     }
    # else:
    #     save_dict = {
    #         'iters': epoch,
    #         'state_dict': net.state_dict(),
    #         'lr': lr
    #     }
    torch.save(save_dict, load_path)
