import json
import random
import xlsxwriter
import os
set = 'train'
# sub_set = 'train'
file = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split_resample/{}_0108.json'.format(set)
import shutil
info_dict = json.load(open(file, 'r'))
video_name = [k for k,_ in info_dict.items()]
selected = random.sample(video_name, 130)



workbook = xlsxwriter.Workbook('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split_resample/ori={}_stem_selected.xlsx'.format(set))
worksheet = workbook.add_worksheet()
root = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/Images'
dest_root = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/Images_stem_selected/{}'.format(set)
if not os.path.exists(dest_root):
    os.mkdir(dest_root)

row = 1
col = 0


for each in (selected):
    src_path = root + "/" + each
    path_save = dest_root + "/" + each
    # if not os.path.exists(path_save):
    #     os.makedirs(path_save)

    # shutil.copy(src_path, path_save)


    shutil.copytree(src_path, path_save)
    worksheet.write(row, col,     each)
    row += 1


workbook.close()


