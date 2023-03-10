# -*- coding:utf-8 -*-
from PIL import Image
from glob import glob
import pandas as pd
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening, convex_hull_image
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
# import pydicom
import scipy.misc
import numpy as np
# import pydicom
# from shutil import copy
# import shutil
# import pydicom
# import SimpleITK as sitk

np.set_printoptions(threshold=1e6)
from skimage import measure, morphology
import zipfile
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, make_response
import warnings


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def Delete_File_Dir(dirName, flag=True):
    # if flag:
    # 	dirName = unicode(dirName, "utf8")
    if os.path.isfile(dirName):
        try:
            os.remove(dirName)
        except:
            pass
    elif os.path.isdir(dirName):
        for item in os.listdir(dirName):
            tf = os.path.join(dirName, item)
            Delete_File_Dir(tf, False)
        try:
            os.rmdir(dirName)
        except:
            pass


executor = ThreadPoolExecutor()
app = Flask(__name__)


@app.route('/test', methods=['GET', 'POST'])
def handle_param():
    if request.method == 'POST':
        try:
            json_zip = request.files['file']
            send = "ok"
            json_zip_name = json_zip.filename
            zip_name = json_zip_name[:-4]
            path = "./files/zip/" + json_zip_name
            json_zip.save(path)
            # new_datas =
            # datas = fusion_json(path,zip_name)

        except:
            send = "false"
    else:
        return "GET not supported, use POST instead"

    return fusion_json(path, zip_name)


'''将所有json的信息以列表提出'''


def fusion_json(path, zip_name):
    save_json_dir = "./files/original/"
    # print(save_json_dir)
    # if not os.path.exists(save_json_dir):
    # 	os.mkdir(save_json_dir)
    unzip_file(path, save_json_dir)
    # origin_path =
    original_path = os.path.join(save_json_dir, zip_name)
    man_path = os.path.join(original_path, "man")
    ai_path = os.path.join(original_path, "ai")

    man_path_list = os.listdir(man_path)
    ai_path_list = os.listdir(ai_path)

    file_len = len(man_path_list)

    recall = 0
    precision = 0
    F1 = 0

    man_list_save = []
    ai_list_save = []
    for man_file_names in man_path_list:
        man_json_path = os.path.join(man_path, man_file_names)
        ai_json_path = os.path.join(ai_path, man_file_names)
        print(man_file_names)

        '''获取每个标注的信息''''''坐标信息的list,层的帧数，id编号和中心点信息'''
        '''人工'''
        with open(man_json_path, "r", encoding="gb2312")as fp:
            json_data_man = json.load(fp)
            # man_list_save.append(1)
        man_list_save.append(json_data_man['classification'])
            # for i in range(len(json_data_man["markInfo"]["RectangleRoi"])):
            #     single_save_man = [list(json_data_man["markInfo"]["RectangleRoi"][i]["handles"].values()),
            #                        json_data_man["markInfo"]["RectangleRoi"][i]["frameNumber"],
            #                        json_data_man["markInfo"]["RectangleRoi"][i]["_id"],
            #                        json_data_man["markInfo"]["RectangleRoi"][i]["imagePosition"]]
            #     man_list_save.append(single_save_man)

        # print(man_list_save)

        '''ai'''
        with open(ai_json_path, "r", encoding="gb2312")as fp:
            json_data_ai = json.load(fp)
        ai_list_save.append(json_data_ai['classification'])

            # for i in range(len(json_data_ai["markInfo"]["RectangleRoi"])):
            #     single_save_ai = [list(json_data_ai["markInfo"]["RectangleRoi"][i]["handles"].values()),
            #                       json_data_ai["markInfo"]["RectangleRoi"][i]["frameNumber"],
            #                       json_data_ai["markInfo"]["RectangleRoi"][i]["_id"],
            #                       json_data_ai["markInfo"]["RectangleRoi"][i]["imagePosition"]]
            #     ai_list_save.append(single_save_ai)
        # print(ai_list_save)

        # '''提取中心点信息'''
        # man_point = []
        # ai_point = []
        #
        # '''寻找人工的中心点'''
        # for man_info in man_list_save:
        #     man_x = man_info[0][4]["x"]
        #     man_y = man_info[0][4]["y"]
        #     man_z = man_info[1]
        #     man_x_plus_y = man_x + man_y
        #     man_temp_save = [man_x, man_y, man_z, man_x_plus_y]
        #     man_point.append(man_temp_save)
        # '''根据x+y的值进行相似点的排序'''
        # # print(man_point)
        # man_point = sorted(man_point, key=lambda x: (x[3]))
        # print(man_point)
        #
        # man_point_save = []
        # man_temp_save = []
        # for i in range(len(man_point) - 1):
        #     if abs(man_point[i][3] - man_point[i + 1][3]) <= 10 and abs(
        #             man_point[i][0] - man_point[i + 1][0]) < 10 and abs(man_point[i][1] - man_point[i + 1][1]) < 10:
        #         man_temp_save.append(man_point[i])
        #         if i == len(man_point) - 2:
        #             man_temp_save.append(man_point[-1])
        #             man_point_save.append(man_temp_save)
        #     else:
        #         man_temp_save.append(man_point[i])
        #         man_point_save.append(man_temp_save)
        #         man_temp_save = []
        #         if i == len(man_point) - 2:
        #             man_temp_save.append(man_point[-1])
        #             man_point_save.append(man_temp_save)
        # print(man_point_save)
        #
        # '''中心点融合最终确认'''
        # man_final_center_point = []
        # for man_item in man_point_save:
        #     if len(man_item) == 1:
        #         man_final_center_point.append(man_item[0])
        #     else:
        #         x_temp = 0
        #         y_temp = 0
        #         z_temp = 0
        #         similar_temp = 0
        #         for i in range(len(man_item)):
        #             x_temp = x_temp + man_item[i][0]
        #             y_temp = y_temp + man_item[i][1]
        #             z_temp = z_temp + man_item[i][2]
        #             similar_temp = similar_temp + man_item[i][3]
        #         x_temp = x_temp / len(man_item)
        #         y_temp = y_temp / len(man_item)
        #         z_temp = z_temp / len(man_item)
        #         similar_temp = similar_temp / len(man_item)
        #         temp_save = [x_temp, y_temp, z_temp, similar_temp]
        #         man_final_center_point.append(temp_save)
        # print(man_final_center_point)
        #
        # "寻找AI的中心点"
        #
        # for ai_info in ai_list_save:
        #     ai_x = ai_info[0][4]["x"]
        #     ai_y = ai_info[0][4]["y"]
        #     ai_z = ai_info[1]
        #     ai_x_plus_y = ai_x + ai_y
        #     ai_temp_save = [ai_x, ai_y, ai_z, ai_x_plus_y]
        #     ai_point.append(ai_temp_save)
        # '''根据x+y的值进行相似点的排序'''
        # ai_point = sorted(ai_point, key=lambda x: (x[3]))
        # # print(ai_point)
        #
        # ai_point_save = []
        # ai_temp_save = []
        # for i in range(len(ai_point) - 1):
        #     if abs(ai_point[i][3] - ai_point[i + 1][3]) <= 10 and abs(ai_point[i][0] - ai_point[i + 1][0]) < 10 and abs(
        #             ai_point[i][1] - ai_point[i + 1][1]) < 10:
        #         ai_temp_save.append(ai_point[i])
        #         if i == len(ai_point) - 2:
        #             ai_temp_save.append(ai_point[-1])
        #             ai_point_save.append(ai_temp_save)
        #     else:
        #         ai_temp_save.append(ai_point[i])
        #         ai_point_save.append(ai_temp_save)
        #         ai_temp_save = []
        #         if i == len(ai_point) - 2:
        #             ai_temp_save.append(ai_point[-1])
        #             ai_point_save.append(ai_temp_save)
        # # print(ai_point_save)
        #
        # ai_final_center_point = []
        # for ai_item in ai_point_save:
        #     if len(ai_item) == 1:
        #         ai_final_center_point.append(ai_item[0])
        #     else:
        #         x_temp = 0
        #         y_temp = 0
        #         z_temp = 0
        #         similar_temp = 0
        #         for i in range(len(ai_item)):
        #             x_temp = x_temp + ai_item[i][0]
        #             y_temp = y_temp + ai_item[i][1]
        #             z_temp = z_temp + ai_item[i][2]
        #             similar_temp = similar_temp + ai_item[i][3]
        #         x_temp = x_temp / len(ai_item)
        #         y_temp = y_temp / len(ai_item)
        #         z_temp = z_temp / len(ai_item)
        #         similar_temp = similar_temp / len(ai_item)
        #         temp_save = [x_temp, y_temp, z_temp, similar_temp]
        #         ai_final_center_point.append(temp_save)
        # # print(ai_final_center_point)

        '''开始匹配'''
        # TP = 0
        # FP = 0
        # FN = 0
        # TN = 0
        # # tp = 0
        # TP =
    man_list_save = np.array(man_list_save)  # as GT
    ai_list_save = np.array(ai_list_save)  # as prediction

    TP = ((man_list_save == 1) * (ai_list_save == 1)).sum()
    TN = ((man_list_save == 0) * (ai_list_save == 0)).sum()
    FP = ((man_list_save == 0) * (ai_list_save == 1)).sum()
    FN = ((man_list_save == 1) * (ai_list_save == 0)).sum()
    #     # for ai_items in ai_final_center_point:
    #     #     ais_x = ai_items[0]
    #     #     ais_y = ai_items[1]
    #     #     ais_z = ai_items[2]
    #     #     ais_similar = ai_items[3]
    #     #     temp = TP
    #     #     for man_items in man_final_center_point:
    #     #         # print(man_items)
    #     #         mans_x = man_items[0]
    #     #         mans_y = man_items[1]
    #     #         mans_z = man_items[2]
    #     #         mans_similar = man_items[3]
    #     #
    #     #         if abs(mans_similar - ais_similar) <= 10 and abs(mans_x - ais_x) < 10 and (mans_y - ais_y) < 10 and (
    #     #                 mans_z - ais_z) < 5:
    #     #             TP = TP + 1
    #     #     if temp == TP:
    #     #         FP = FP + 1
    #     #
    #     # for man_itemss in man_final_center_point:
    #     #     manss_x = man_itemss[0]
    #     #     manss_y = man_itemss[1]
    #     #     manss_z = man_itemss[2]
    #     #     manss_similar = man_itemss[3]
    #     #     temp = tp
    #     #     for ai_itemss in ai_final_center_point:
    #     #         aiss_x = ai_itemss[0]
    #     #         aiss_y = ai_itemss[1]
    #     #         aiss_z = ai_itemss[2]
    #     #         aiss_similar = ai_itemss[3]
    #     #         if abs(manss_similar - aiss_similar) <= 10 and abs(manss_x - aiss_x) < 10 and (
    #     #                 manss_y - aiss_y) < 10 and (manss_z - aiss_z) < 5:
    #     #             tp = tp + 1
    #     #     if temp == tp:
    #     #         FN = FN + 1
    #
    #     # print(TP,FP,FN)
    #     if (TP + FN) == 0:
    #         recall = 0 + recall
    #     else:
    #         recall = TP / (TP + FN) + recall
    #     if (TP + FP) == 0:
    #         precision = 0 + precision
    #     else:
    #         precision = TP / (TP + FP) + precision
    #     if precision + recall == 0:
    #         F1 = 0 + F1
    #     else:
    #         F1 = (2 * precision * recall) / (precision + recall) + F1
    # recall = recall / file_len
    # precision = precision / file_len
    # F1 = F1 / file_len
    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = (2 * precision * recall) / (precision + recall)
    recall = str(recall)
    precision = str(precision)
    F1 = str(F1)

    print(recall, precision, F1)
    json_info = {
        "recall": recall,
        "precision": precision,
        "F1": F1
    }
    new_data = json.dumps(json_info)
    return new_data


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=9990,
        debug=True
    )
#
#
