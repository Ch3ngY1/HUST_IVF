
import os.path
import json

import shutil

root='/data/mxj/IVF_HUST/first_travel_VideoImage'
filename='valid.json'
image_tmpl = "{:d}.jpg"
list_file = os.path.join(root, 'data_split', filename)
info_dict = json.load(open(list_file, 'r'))
video_list = [x.strip() for x in info_dict]
a = 1
sub_path = 'Images'
dest_root = '/data2/chengyi/dataset/HUST_val'
for each_video in video_list:
    file_name = os.path.join(root, sub_path, each_video)
    dest = os.path.join(dest_root, sub_path, each_video)
    shutil.copytree(file_name, dest)  # 拷贝目录，bbc若存在将报错
    # os.system('xcopy %s %s /s' % (file_name, dest))  # 拷目录，/s 复制非空的目录和子目录。
