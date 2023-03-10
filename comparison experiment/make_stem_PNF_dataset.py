

import pandas as pd
import json
import numpy as np
import math


set = 'train'

file1 = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/{}_PNF_onecell.json'.format(set)
# 用于知道哪些图片是t1的
info_dict1 = json.load(open(file1, 'r'))

file2 = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/train.json'
# 用于视频可用帧有哪些
info_dict2 = json.load(open(file2, 'r'))


# video_name = [k for k,_ in info_dict.items()]
label_list = ['0:PN','1:NON-PN']
df = pd.read_excel('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/stem-PNF-{}.xlsx'.format(set))
# 知道哪些图片是pna pnf的

json_file_out = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/{}_PNF_final.json'.format(set)
data=df.values
out_dir = {}
for each in data:
    # index_map = {}
    file, pna, pnf = each
    present2 = info_dict2[file]
    present1 = info_dict1[file]

    mapping_indexes = np.array(present2['mapping_indexes'])
    # 所有的cell-1 图像
    start_index = np.where(mapping_indexes == pna)
    end_index = np.where(mapping_indexes == pnf)
    for img in mapping_indexes:
        if img > present1:
            break
        name = file+'_'+str(img)
        if img < start_index or img > end_index:
            label = label_list[1]
        else:
            label = label_list[0]
        out_dir[name] = label

my_out = json.dumps(out_dir)
f2 = open(json_file_out, 'w')
f2.write(my_out)
f2.close()






# print("获取到所有的值:\n{}".format(data))