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

# constants
WINDOW_NAME = "COCO detections"
# def unzip_file(zip_src, dst_dir):
#     r = zipfile.is_zipfile(zip_src)
#     if r:
#         fz = zipfile.ZipFile(zip_src, 'r')
#         for file in fz.namelist():
#             fz.extract(file, dst_dir)
#     else:
#         print('This is not zip')

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

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', '/data/lxc/output/cervix_det/faster_r50_fpn/hsil/acid/model_0074999.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser
app = Flask(__name__,static_folder='/data/db/cervix_det/save', static_url_path='/file')
# app = Flask(__name__)
@app.route('/detection_param', methods=['GET', 'POST'])

def handle_param():
    if request.method == 'POST':
        try:
            img = request.files['file']
            pic_name = img.filename
            path = "save/"+pic_name
            # print(path)
        except:
            result = "POST参数key name错误，示例：\n" \
                     "key: 'url'，value: 图片url"
            # print()
        else:
            img.save(path)
            img_url = path
            img = read_image(img_url, format="BGR")
            predictions, visualized_output = demo.run_on_image(img)
            # out_filename = "/data/db/cervix_det/result/"
            value = len(predictions["instances"])
            pic_root = "http://192.168.0.240:9999/file/det_acid_"+pic_name+"?="+str(value)
            visualized_output.save("/data/db/cervix_det/save/det_acid_"+pic_name)



            ''''''
            json_info = {
                "modalNo": testfilelist[0],
                "aiJson": global_det,

            }
            print(json_info)
            head = {"Content-Type": "application/json; charset=UTF-8", 'Connection': 'close'}
            data = json.dumps(json_info)
            # data = deal_json_invaild(data)
            print(data)
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
    demo = VisualizationDemo(cfg)
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
