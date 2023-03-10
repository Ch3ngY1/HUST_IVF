import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, utils
from utils.config import cfg
import time
from sklearn import metrics
import torch.nn as nn
import json
'''
CUDA_VISIBLE_DEVICES=6 python3 visual_frame_choose.py test --net Ada --tag much_seg -tf -rs --modality Ada --num_segments 700 --lr 0.0003 -ld 25 --epoch 75 --loss Cross_Entropy --bs 1

'''


def feature_extraction(model, img):
    # image = [N, 128, 1 ,224, 224]
    # net: 2D CNN
    b, f, c, h, w = img.shape
    img = img.view(b * f, c, h, w)  # --> (128 * N) * 1 * 224 * 224

    feature = model(img)
    feature = feature.view(b, f, -1)
    return feature

torch.nn.Module.dump_patches = True
lossfc = nn.MSELoss()
args = my_parse.parse_args()
GPU = 1
net_size = 96
# checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Dec16_05-48-34_Ada_Ada/best_val_acc.pth',
#                         map_location='cpu')
checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan04_05-35-00_Ada_Ada_randfix_tau_0.5/best_val_acc.pth')
if GPU:
    net = checkpoint['net'].feature.cuda()
else:
    net = checkpoint['net'].feature

# ==========================================
# TODO: for loop
# TODO: 做成函数
# filename = 'train'
filename = 'valid'
print('================= ' + filename + ' =================')
# ==========================================
val_dataloader = my_dataloader.my_dataloader_visual(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                    filename='{}.json'.format(filename), args=args, train=False)
json_file = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/{}.json'.format(filename)
# TODO：改为时间的
json_file_out = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split_resample/{}_0108.json'.format(filename)
f = open(json_file, 'r')
content = f.read()
out_dir = json.loads(content)
f.close()


test_predict_list, test_target_list = [], []
net.eval()
val_loss_total = 0
for batch_idx, (blobs, targets, videoname, indices) in enumerate(val_dataloader):

    utils.progress_bar(batch_idx, len(val_dataloader))

    with torch.no_grad():
        cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)

        blobs[0] = utils.img_data_to_variable(blobs[0], gpu=GPU)

        img = blobs[0]
        position = blobs[1]
        feature = []

        for a in range(args.num_segments // net_size):
            feature.append(feature_extraction(net, img[:, a * net_size:net_size * (a + 1), ...]))

        if args.num_segments // net_size * net_size != args.num_segments:
            feature.append(feature_extraction(net, img[:, args.num_segments // net_size * net_size:, ...]))
        feature = torch.cat(feature, dim=1)
        theshold = 0.03  # adjustable
        diff = [lossfc(x1, x2).item() - theshold for x1, x2 in
                zip(feature[0, 0:args.num_segments - 1, :], feature[0, 1:args.num_segments, :])]
        # print(diff)
        # print(videoname)
        # print(indices)

        sample = torch.zeros(len(diff))
        win = 10  # adjustable
        for i in range(len(diff)):
            if i < win:
                now = torch.cat([torch.zeros(win - i), torch.tensor(diff[0:i + win + 1])])
            elif i >= len(diff) - win:
                now = torch.cat([torch.tensor(diff[i:]), torch.zeros(-len(diff) + i + 2 * win + 1)])
            else:
                now = diff[i - win:i + win + 1]

            if now[win] >= max(now) and now[win] != 0:
                sample[i] = 1
        remain = int(32 - sum(sample).item())

        if remain < 0:
            remain = -remain
            del_front = remain // 2
            del_back = remain - del_front
            index = torch.nonzero(sample)
            out_index = index[del_front:-del_back]
        elif remain > 0:
            interval = len(diff) // remain
            more_index = list(range(interval//2, len(diff), interval))
            index = torch.nonzero(sample)
            more_index = [x+1 if x in index else x for x in more_index]
            sample[more_index] = 1
            out_index = torch.nonzero(sample)
        else:
            out_index = torch.nonzero(sample)
        sample_index = out_index.tolist()
        # out_dir[videoname] = sample_index
        sample_index = [x[0] for x in sample_index]
        if len(sample_index) != 32:
            print('value error')
            raise ValueError
        out_dir[videoname[0]]['mapping_indexes'] = [out_dir[videoname[0]]['mapping_indexes'][x] for x in sample_index]
        # [out_dir[videoname[0]]['mapping_indexes'][x] for x in sample_index]
my_out = json.dumps(out_dir)
f2 = open(json_file_out, 'w')
f2.write(my_out)
f2.close()
# save json file
    # print(feature.shape)
