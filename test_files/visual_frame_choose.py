import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
import torch
import numpy as np

sys.path.append('../')
from utils import my_parse, my_dataloader, utils
from utils.config import cfg
import time
from sklearn import metrics
import torch.nn as nn
import json

'''
CUDA_VISIBLE_DEVICES=6 python3 visual_frame_choose.py test --net Ada --tag much_seg -tf -rs --modality Ada --num_segments 700 --lr 0.0003 -ld 25 --epoch 75 --loss Cross_Entropy --bs 1

'''


def choose(diff, win=10, dest=32):
    # ===================================== Choose Frame =====================================
    sample = torch.zeros(len(diff))
    # win = 10  # adjustable
    for i in range(len(diff)):
        if i < win:
            now = torch.cat([torch.zeros(win - i), torch.tensor(diff[0:i + win + 1])])
        elif i >= len(diff) - win:
            now = torch.cat([torch.tensor(diff[i:]), torch.zeros(-len(diff) + i + 2 * win + 1)])
        else:
            now = diff[i - win:i + win + 1]

        if now[win] >= max(now) and now[win] != 0:
            sample[i] = 1
    # tmp_index = np.array(sample)
    # tmp = np.nonzero(tmp_index)[0]
    #
    # remain = int(dest - len(tmp))
    return sample


def interpolation_base(remain, sample):
    # ===================================== Interpolation =====================================
    # 少的情况是采样，多的画直接从头尾剪（不合理）
    if remain < 0:
        remain = -remain
        del_front = remain // 2
        del_back = remain - del_front
        index = torch.nonzero(sample)
        out_index = index[del_front:-del_back]
    elif remain > 0:
        interval = len(diff) // remain
        more_index = list(range(interval, len(diff), interval))
        index = torch.nonzero(sample)
        more_index = [x + 1 if x in index else x for x in more_index]
        sample[more_index] = 1
        out_index = torch.nonzero(sample)
    else:
        out_index = torch.nonzero(sample)
    return out_index


def interpolation_interval(sample_original,len_total, offset, dest):
    # ===================================== Interpolation =====================================
    tmp_index_otiginal = np.array(sample_original)
    tmp_original = np.nonzero(tmp_index_otiginal)[0]
    remain_original = int(dest - len(tmp_original))
    sample = sample_original
    remain = remain_original
    # 少的情况从最大的interval之间重新取，多的情况删去最密集的两个中间帧
    if remain < 0:
        for i in range(-remain):
            remain = -remain
            tmp_index = np.array(sample)
            tmp = np.nonzero(tmp_index)[0]
            tmp_diff = [tmp[ii+2] - tmp[ii] for ii in range(len(tmp)-2)]
            # tmp = tmp[2:] - tmp[0:-2]
            tmp_min_index = np.where(tmp_diff == np.min(tmp_diff))
            del_index = tmp[tmp_min_index[0] + 1] if len(tmp_min_index[0])==1 else tmp[tmp_min_index[0][0] + 1]
            sample[del_index] = 0
        out_index = torch.nonzero(sample)

    elif remain > 0:
        for ii in range(remain):
            tmp_index = np.array(sample)
            l1 = np.append(np.array([0]),np.nonzero(tmp_index)[0])
            l2 = np.append(np.nonzero(tmp_index)[0], np.array([len_total-1]))
            ll = l2-l1
            tmp_max_index = np.where(ll == np.max(ll))
            # if
            try:
                tmp_max_low = l1[tmp_max_index] if len(tmp_max_index[0])==1 else l1[tmp_max_index[0][0]]
                tmp_max_high = l2[tmp_max_index] if len(tmp_max_index[0])==1 else l2[tmp_max_index[0][0]]
            except IndexError:
                print('?')
            additional_index = (tmp_max_low+tmp_max_high)//2
            sample[additional_index] = 1
        # interval = len_total // remain
        # more_index = list(range(interval, len_total, interval))
        # index = torch.nonzero(sample)
        # more_index = [x + 1 if x in index else x for x in more_index]
        # sample[more_index] = 1
        out_index = torch.nonzero(sample)
    else:
        out_index = torch.nonzero(sample)
    if len(out_index) != dest:
        print('?')
    return out_index+offset


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
checkpoint = torch.load(
    '/data2/chengyi/myproject/Savings/save_models/Jan12_08-09-25_Ada_Ada_adatest/best_val_acc.pth') # acc=0.7
if GPU:
    net = checkpoint['net'].feature.cuda()
else:
    net = checkpoint['net'].feature
# ==========================================
# TODO: for loop
# TODO: 做成函数
filename = 'train'
# filename = 'valid'
# filename = 'test'
print('================= ' + filename + ' =================')
# ==========================================
val_dataloader = my_dataloader.my_dataloader_visual(root='/data/mxj/IVF_HUST/first_travel_VideoImage',
                                                    filename='{}.json'.format(filename), args=args, train=False)
json_file = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/{}.json'.format(filename)
# TODO：改为时间的
json_file_out = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split_resample/{}_0114.json'.format(filename)
f = open(json_file, 'r')
content = f.read()
out_dir = json.loads(content)
f.close()

test_predict_list, test_target_list = [], []
net.eval()
val_loss_total = 0
for batch_idx, (blobs, targets, videoname, indices) in enumerate(val_dataloader):

    utils.progress_bar(batch_idx, len(val_dataloader))
    # print(videoname)
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

        num_seg = 7
        interval = len(diff) // num_seg
        # dest_num_list = [32//num_seg]*num_seg

        dest_num_list = [
            32 // num_seg + 1 if divmod(i, (num_seg - 1) / (divmod(32, num_seg)[1] - 1))[1] == 0 else 32 // num_seg for
            i in range(num_seg)]
        out_index = []
        for i in range(num_seg):
            if i == num_seg - 1:
                partial_diff = diff[interval * i:]
            else:
                partial_diff = diff[interval * i:interval * (i + 1)]
            sample = choose(diff=partial_diff, win=10)
            after_interpolation = interpolation_interval(sample_original=sample, len_total=len(partial_diff), offset = i*len(sample), dest=dest_num_list[i])
            out_index.extend(after_interpolation)

        sample_index = torch.stack(out_index).tolist()
        if len(sample_index) != 32:
            print('?????????????????????????????')
        # out_dir[videoname] = sample_index
        out_dir[videoname[0]]['mapping_indexes'] = [out_dir[videoname[0]]['mapping_indexes'][each[0]] for each in sample_index]
my_out = json.dumps(out_dir)
f2 = open(json_file_out, 'w')
f2.write(my_out)
f2.close()
# save json file
# print(feature.shape)
