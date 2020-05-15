from __future__ import division, print_function

import tensorflow as tf
import numpy as np 
import argparse
import cv2
import time
import shapely.geometry as sg
import shapely.ops as so


from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
from shapely.geometry import Polygon

from my_model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("input_video", type=str,
                    help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with 'new_size'")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the video detection results.")
args = parser.parse_args()

args.anchor = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
arg.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
vid = cv2.VideoCapture(args.input_video)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

distance = 15
def RoI_regularize(prev_boxes, frame_id):
    mv = np.loadtxt('mv{}'.format(frame_id))
    r = sg.box(0, 0, 0, 0)
    if len(prev_boxes) > 0:
        for box in prev_boxes:
            r1 = sg.box(box[0], box[1], box[2], box[3])
            r = r.union(r1)
    if len(mv) > 0:
        mv = mv[:, 1:]
        for box in mv:
            r1 = sg.box(box[0], box[1], box[2], box[3])
            r =r.union(r1)
    res = []
    if r.geom_type == 'MultiPolgon':
        polysize = len(r.geoms)
        for i in range(polysize):
            ploy = r.geoms[i]
            if len(res) == 0:
                res = [ploy.bounds[0], ploy.bounds[1], ploy.bounds[2], ploy.bounds[3]]
            else:
                res = [res, [ploy.bounds[0], ploy.bounds[1], ploy.bounds[2], ploy.bounds[3]]]
    else:
        if r.area != 0:
            res = [r.bounds[0], r.bounds[1], r.bounds[2], r.bounds[3]]
    if len(res) != 0:
        res = np.reshape(res, [-1, 4])
    return res
with tf.Session() as sess:


    for i in range(video_frame_cnt):
        ret, img_ori = vid.read()
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori)
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        if i % distance == 0:
            # call RoI_regularize for next distance frames to get the union region(R)
            # need load txt file to get "prev_boxes" in RoI_regularize func
            r = sg.box(0, 0, 0, 0)
            for j in range(i+1, i+distance):
                res = (prev, j)
                if res != []:
                    for box in res:
                        temp = sg.box(box[0], box[1], box[2], box[3])
                        r = r.union(temp)
            saved_region = [r.bounds[0], r.bounds[1], r.bounds[2], r.bounds[3]]
            
            # call model 1 here, return boxes and a lot of layers for R
            
        else:
            # call model 1 here, with the small input image and a lot of layers




            
