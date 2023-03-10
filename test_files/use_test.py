import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel
from utils.config import cfg
import time
from sklearn import metrics

'''
瞎猜acc=0.62
暂时将我们的模型命名为Ada
'''
if __name__ == '__main__':
    args = my_parse.parse_args()

    GPU = 0


    checkpoint = torch.load('/data2/chengyi/myproject/save_models/Dec16_05-48-34_Ada_Ada/best_val_acc.pth', map_location='cpu')
    net = checkpoint['net']
    # net.froze_bn()

    val_dataloader = my_dataloader.my_dataloader_visual(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                 filename='valid.json', args=args, train=False)

    # =========================================== validation ===========================================

    net.eval()
    for batch_idx, (blobs, targets, videoname, indices) in enumerate(val_dataloader):
        cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
        print(videoname)
        print(indices)
        blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)
        with torch.no_grad():
            if args.net == 'Ada':
                pred_list = net(blobs)
