import json
import random
#
# # a = json.load(open('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/train.json', 'r'))
# valid_original = json.load(open('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/valid.json', 'r'))
test_original = json.load(open('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/test_resample.json', 'r'))
# train_original = json.load(open('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/train_resample_new.json', 'r'))
# # a_key = [x.strip() for x in a]
# valid_video = [x.strip() for x in valid_original]
test_video = [x.strip() for x in test_original]
# train_video = [x.strip() for x in train_original]
#
#
# print(len(list(set(test_video).intersection(set(train_video))))) # #列用集合的取交集方法
# print(len(list(set(valid_video).intersection(set(train_video))))) # #列用集合的取交集方法
# print(len(list(set(test_video).intersection(set(valid_video))))) # #列用集合的取交集方法
#
# intersected = list(set(test_video).intersection(set(train_video)))
#
# # testandtrain = train_original
# # testandtrain.update(test_original)
# #
# #
# # # for a in valid_original:
# # #     print(a)
# # num0 = 127
# # num1 = 330-127
# #
train1 = []
train0 = []
for each in test_video:
    if test_original[each]['label'] == 1:
        train1.append(each)
    else:
        train0.append(each)
print(len(train1)/(len(train1)+len(train0)))
# random.shuffle(train1)
# # random.shuffle(train0)
# #
# # test1 = train1[:num1]
# # test0 = train0[:num0]
# #
# # train1 = train1[num1:]
# # train0 = train0[num0:]
# #
# # test_key = test0 + test1
# # train_key = train0 + train1 + test_video
# #
# # test_data_set = {}
# # json_file_out_test = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/test_resample.json'
# # for each in test_key:
# #     test_data_set[each] = testandtrain[each]
# #
# # my_out = json.dumps(test_data_set)
# # f2 = open(json_file_out_test, 'w')
# # f2.write(my_out)
# # f2.close()
# #
# #
# train_data_set = {}
# json_file_out_train = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/train_resample_new.json'
# for each in intersected:
#     del train_original[each]  # 删除键是'Name'的条目
#
# print(len([x.strip() for x in train_original]))
#
#
#
# my_out = json.dumps(train_original)
# f2 = open(json_file_out_train, 'w')
# f2.write(my_out)
# f2.close()

import torch
import math
import numpy as np
max_len = 2048
indexes = np.random.randint(0, 800, size=32).tolist()
indexes.sort()


# dim = len(indexes)
dim = 850
pe = torch.zeros(max_len, dim)
position = torch.arange(0, max_len).unsqueeze(1)
div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
pe[:, 0::2] = torch.sin(position.float() * div_term)
pe[:, 1::2] = torch.cos(position.float() * div_term)
# pe.shape = 2048(LSTM length) * 128(data length)
pe = pe.transpose(1, 0)
pefinal = torch.stack([pe[x] for x in indexes], dim=0)
print(pefinal.shape)

