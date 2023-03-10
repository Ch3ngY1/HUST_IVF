# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import sys
sys.path.append('/data/db/cervix_det')
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from flask import Flask, render_template, request,jsonify
import requests
from io import BytesIO
import json
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
import math

# constants
WINDOW_NAME = "COCO detections"





class MY_DEMO():
    def __init__(self, cpkt):
        checkpoint = torch.load(cpkt)
        self.net = checkpoint['net'].cuda()

    def run(self, img):
        pred_list = self.net(img)
        final_pred = pred_list[1]
        # if args.adamode == 'bi':
        #     final_pred = pred_list[1] + pred_list[4]
        pred = final_pred[-1].data.max(1)[1]
        return pred

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, num_segments, size, div=True, modality='Ada', new_length=1):
        self.div = div
        self.seg = num_segments
        self.size = size
        self.modality = modality
        self.new_length = new_length

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            if self.modality == 'Ada':
                seg, _, _ = img.shape
                # TODO: 要同时给all和3frame用
                img = img.view(1, seg, self.size, self.size)

            else:
                img = img.view(2 * self.new_length, self.seg, self.size, self.size)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img.float().div(255) if self.div else img.float()

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll
        # self.dim = dim

    def __call__(self, img_group):
        # if self.dim == 3:
        #     return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

class Sample_Images():
    def __init__(self, file_dir, pos = None):
        self.positionmode = pos
        self.file_dir = file_dir
        self.sample_interval = 1
        self.num_segments = 32
        pre_process = Stack(roll=False)
        augmentation = torchvision.transforms.Compose([GroupScale(int(224 * 256 // 224)),
                                                       GroupCenterCrop(224)])
        self.transform = torchvision.transforms.Compose([
            # NormalTransform(),
            augmentation,
            pre_process,
            ToTorchFormatTensor(32, 224, div=True,
                                modality='Ada', new_length=1),
        ])

        self.new_length = 1


    def _load_image(self, file_name):
        # print(directory)
        return [Image.open(file_name).convert('L')]


    def _get_val_indices(self, num_frame, num_segments):
        # 采样位置为： 50-750

        # if num_frame > 750:
        #     num_frame = 750

        num_frame = num_frame - 50

        if num_frame > num_segments + self.new_length * self.sample_interval - 1:
            tick = (num_frame - self.new_length * self.sample_interval + 1) / float(num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))

        return offsets + 50

    def get(self):
        img_dir = self.file_dir
        files = os.listdir(img_dir)
        num_frame = len(files)
        indices = self._get_val_indices(num_frame, self.num_segments)
        images = list()
        indexes = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:
                    sub_path = 'Images'
                    # TODO: video_name
                    seg_imgs = self._load_image(os.path.join(self.file_dir, img_dir, str(i)+'.jpg'))
                except IndexError:
                    # print(video_name, len(record['mapping_indexes']), p)
                    # print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                indexes.append(p)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < num_frame:
                    p += self.sample_interval

        process_data = self.transform(images)
        positional_encoding = self.positional_encoding(indexes)
        # if self.positionmode:
        return process_data, positional_encoding


    def positional_encoding(self, indexes, max_len=2048):
        '''
        :param indexes: dtype = int, max_len: int
        max_len: length of LSTM input
        dim = indexes length

        :return:
        max_len * dim
        '''
        if self.positionmode is None:
            return None
        dim = 850
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe.shape = 2048(LSTM length) * 128(data length)
        pe = pe.transpose(1, 0)
        pe = torch.stack([pe[x] for x in indexes], dim=0)
        return pe


    def classify_collate(self, batch):
        '''
        :param batch: [imgs, indexes, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        [img, position], label
        '''

        if isinstance(batch[0][1], torch.Tensor):
            positional_encoding = torch.stack([x[1] for x in batch])
        elif isinstance(batch[0][1], list):
            positional_encoding = [x[1] for x in batch]
        else:
            positional_encoding = None

        imgs = [x[0] for x in batch]
        # return [torch.stack(imgs, 0).transpose(2, 1), torch.stack(positional_encoding)], np.asarray(labels).astype(
        #     np.int64)
        return [torch.stack(imgs, 0).transpose(2, 1), positional_encoding]




def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/data/lxc/output/cervix_det/faster_r50_fpn/hsil/acid/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images",default="pic/00510673_2016-12-29_2.jpg")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
        default="result"
    )

    # parser.add_argument(
    #     "--confidence-threshold",
    #     type=float,
    #     default=0.5,
    #     help="Minimum score for instance predictions to be shown",
    # )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # Jan03_04-37-55_Ada_Ada_del_random_fix_initialized
        default=['MODEL.WEIGHTS', '/data2/chengyi/myproject/Savings/save_models/Feb02_10-00-05_Ada_Ada_cat850cpcr/best_val_acc.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser
app = Flask(__name__,static_folder='/data/db/cervix_det/save', static_url_path='/file')
# app = Flask(__name__)
@app.route('/detection_param', methods=['GET', 'POST'])

def handle_param():
    if request.method == 'POST':
        try:
            img_dir = request.files['file']
            dir_name = img_dir.filename
            path = "save/"+ dir_name
            # print(path)
        except:
            result = "POST参数key name错误，示例：\n" \
                     "key: 'url'，value: 图片url"
            # print()
        else:
            # TODO: 保存文件夹
            dir_name.save(path)
            img_url = path
            # TODO:改为读取32张图像
            img_sampler = Sample_Images(img_url)
            img = img_sampler.get()

            # predictions, visualized_output = demo.run_on_image(img)
            #
            # checkpoint = torch.load('/data2/chengyi/myproject/Savings/save_models/Jan03_04-37-55_Ada_Ada_del_random_fix_initialized/best_val_acc.pth')
            # net = checkpoint['net'].cuda()
            # predictions = net(img)

            prediction = demo.run(img)
            # out_filename = "/data/db/cervix_det/result/"
            # value = len(predictions["instances"])

            json_info = {'classification': prediction}


            # json_info = {
            #     "modalNo": testfilelist[0],
            #     "aiJson": global_det,
            #
            # }
            print(json_info)
            head = {"Content-Type": "application/json; charset=UTF-8", 'Connection': 'close'}
            data = json.dumps(json_info)
            # data = deal_json_invaild(data)
            print(data)
            # TODO: 修改URL地址
            r = requests.post(url="http://39.106.229.212:9080/prod-api/orthanc/ai/callback",
                              data=data, headers=head)
            print(r.text)


    else:
        return 'GET not supported, use POST instead'

    return pic_root
            # print((img))
            # img_url = "/data/db/cervix_det/demo/00510673_2016-12-29_2.jpg"
            # print(img_url)
    #     except:
    #         result = "POST参数key name错误，示例：\n" \
    #                  "key: 'url'，value: 图片url"
    #         status = 0
    #     else:
    #         try:
    #             # img = read_image(img_url, format="BGR")
    #
    #         except:
    #             result = '图像下载错误'
    #             status = 0
    #         else:
    #             try:
    #                 # img = read_image(img_url, format="BGR")
    #                 print(2)
    #                 predictions, visualized_output = demo.run_on_image(img)
    #                 out_filename = "/data/db/cervix_det/result"
    #                 # visualized_output.save("/data/db/cervix_det/result/aa.jpg")
    #             except:
    #                 result = 'inference error'
    #                 status = 0
    #             else:
    #                 status = 1
    #
    #
    #     return out_filename  #jsonify(result_dict)
    #
    # else:
    #     return 'GET not supported, use POST instead'
#

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    # demo = VisualizationDemo(cfg)
    #
    # checkpoint = torch.load(args.opts)
    # net = checkpoint['net'].cuda()
    demo = MY_DEMO(args.opts)

    app.run(
        host='0.0.0.0',
        port=9999,
        debug=True
    )

    # if args.input:
    # #     if len(args.input) == 1:
    # #         args.input = glob.glob(os.path.expanduser(args.input[0]))
    # #         assert args.input, "The input path(s) was not found"
    # #     for path in tqdm.tqdm(args.input, disable=not args.output):
    # #         # use PIL, to be consistent with evaluation
    # #         path = "pic/00510673_2016-12-29_2.jpg"
    #         img = read_image(path, format="BGR")
    #         # start_time = time.time()
    #         predictions, visualized_output = demo.run_on_image(img)
    #         # logger.info(
    #         #     "{}: detected {} instances in {:.2f}s".format(
    #         #         path, len(predictions["instances"]), time.time() - start_time
    #         #     )
    #         # )
    #
    #         if args.output:
    #             if os.path.isdir(args.output):
    #                 assert os.path.isdir(args.output), args.output
    #                 out_filename = os.path.join(args.output, os.path.basename(path))
    #             else:
    #                 assert len(args.input) == 1, "Please specify a directory with args.output"
    #                 out_filename = args.output
    #             visualized_output.save(out_filename)

    # elif args.webcam:
    #     assert args.input is None, "Cannot have both --input and --webcam!"
    #     cam = cv2.VideoCapture(0)
    #     for vis in tqdm.tqdm(demo.run_on_video(cam)):
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, vis)
    #         if cv2.waitKey(1) == 27:
    #             break  # esc to quit
    #     cv2.destroyAllWindows()
    # elif args.video_input:
    #     video = cv2.VideoCapture(args.video_input)
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frames_per_second = video.get(cv2.CAP_PROP_FPS)
    #     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     basename = os.path.basename(args.video_input)
    #
    #     if args.output:
    #         if os.path.isdir(args.output):
    #             output_fname = os.path.join(args.output, basename)
    #             output_fname = os.path.splitext(output_fname)[0] + ".mkv"
    #         else:
    #             output_fname = args.output
    #         assert not os.path.isfile(output_fname), output_fname
    #         output_file = cv2.VideoWriter(
    #             filename=output_fname,
    #             # some installation of opencv may not support x264 (due to its license),
    #             # you can try other format (e.g. MPEG)
    #             fourcc=cv2.VideoWriter_fourcc(*"x264"),
    #             fps=float(frames_per_second),
    #             frameSize=(width, height),
    #             isColor=True,
    #         )
    #     assert os.path.isfile(args.video_input)
    #     for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
    #         if args.output:
    #             output_file.write(vis_frame)
    #         else:
    #             cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
    #             cv2.imshow(basename, vis_frame)
    #             if cv2.waitKey(1) == 27:
    #                 break  # esc to quit
    #     video.release()
    #     if args.output:
    #         output_file.release()
    #     else:
    #         cv2.destroyAllWindows()
