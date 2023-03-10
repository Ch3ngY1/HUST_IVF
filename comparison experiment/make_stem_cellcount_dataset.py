# import numpy as np
# data = np.loadtxt(open("/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/ori=valid_stem_selected.csv.csv","rb"),delimiter=",",skiprows=n,usecols=[2,3])

# import pandas as pd
# data = pd.read_csv('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/ori=valid_stem_selected.csv')
# print(data)

# import xlrd
# #打开excel
# wb = xlrd.open_workbook('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/ori=valid_stem_selected.xlsx')
# #按工作簿定位工作表
# sh = wb.sheet_by_name('TestUserLogin')
# print(sh.nrows)#有效数据行数
# print(sh.ncols)#有效数据列数
# print(sh.cell(0,0).value)#输出第一行第一列的值
# print(sh.row_values(0))#输出第一行的所有值
# #将数据和标题组合成字典
# print(dict(zip(sh.row_values(0),sh.row_values(1))))
# #遍历excel，打印所有数据
# for i in range(sh.nrows):
#     print(sh.row_values(i))

import pandas as pd
import json
import numpy as np
import math

set = 'test'

file = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/data_split/{}.json'.format(set)

info_dict = json.load(open(file, 'r'))
# video_name = [k for k,_ in info_dict.items()]
label_list = ['t1','t2','t3','t4','t5+']
df = pd.read_excel('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/ori={}_stem_selected.xlsx'.format(set))
json_file_out = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/{}_.json'.format(set)
data=df.values
out_dir = {}
for each in data:
    # index_map = {}
    file, t2, t3, t4, t5, end = each
    present = info_dict[file]
    if math.isnan(end):
        end = present['mapping_indexes'][-1]

    mapping_indexes = np.array(present['mapping_indexes'])
    index_list = [t2,t3,t4,t5,end]
    for i in range(5):
        if index_list[i] == -1:
            break
        if i == 0:
            start = present['mapping_indexes'][0]
        else:
            start = index_list[i-1]
        end = index_list[i]
        start_index = np.where(mapping_indexes == start)
        end_index = np.where(mapping_indexes == end)
        imgs = mapping_indexes[start_index[0].item():end_index[0].item()]
        for img in imgs:
            name = file+'_'+str(img)
            label = label_list[i]
            out_dir[name] = label

my_out = json.dumps(out_dir)
f2 = open(json_file_out, 'w')
f2.write(my_out)
f2.close()






# print("获取到所有的值:\n{}".format(data))