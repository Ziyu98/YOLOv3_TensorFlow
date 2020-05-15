# coding: utf-8
#

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import shapely.geometry as sg

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model3 import yolov3
#from model2 import yolov3_2

parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("input_video", type=str,
                    help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the video detection results.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

vid = cv2.VideoCapture(args.input_video)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))
layersize = np.zeros(49)
for i in range(49):
    if i <= 0:
        layersize[i] = 416
    elif i <= 2 and i > 0:
        layersize[i] = 208
    elif i <= 5 and i > 2:
        layersize[i] = 104
    elif i <= 14 and i > 5:
        layersize[i] = 52
    elif i <= 23 and i > 14:
        layersize[i] = 26
    elif i <= 35 and i > 23:
        layersize[i] = 13
    elif i <= 42 and i > 35:
        layersize[i] = 26
    elif i <= 48 and i > 42:
        layersize[i] = 52

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
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    """input_n1 = tf.placeholder(tf.float32, [1, 416, 416, None])
    input_n2 = tf.placeholder(tf.float32, [1, 208, 208, None])
    input_n3 = tf.placeholder(tf.float32, [1, 208, 208, None])
    input_n4 = tf.placeholder(tf.float32, [1, 104, 104, None])
    input_n5 = tf.placeholder(tf.float32, [1, 104, 104, None])
    input_n6 = tf.placeholder(tf.float32, [1, 104, 104, None])
    input_n7 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n8 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n9 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n10 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n11 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n12 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n13 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n14 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n15 = tf.placeholder(tf.float32, [1, 52, 52, None])
    input_n16 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n17 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n18 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n19 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n20 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n21 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n22 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n23 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n24 = tf.placeholder(tf.float32, [1, 26, 26, None])
    input_n25 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n26 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n27 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n28 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n29 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n30 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n31 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n32 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n33 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n34 = tf.placeholder(tf.float32, [1, 13, 13, None])
    input_n35 = tf.placeholder(tf.float32, [1, 13, 13, None])"""
    for i in range(1, 50):
        locals()['input_n' + str(i)] = tf.placeholder(tf.float32, [1, layersize[i - 1], layersize[i - 1], None])
    region = tf.placeholder(tf.int32)

    yolo_model = yolov3(args.num_class, args.anchors)
    #yolo_model_2 = yolov3_2(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        feature_map_1, feature_map_2, feature_map_3, net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, net12, net13, net14, net15, net16, net17, net18, net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33, net34, net35, net36, net37, net38, net39, net40, net41, net42, net43, net44, net45, net46, net47, net48, net49 = yolo_model.forward_1(input_data, False)
    #with tf.variable_scope('yolov3'):
        f1, f2, f3 = yolo_model.forward_2(input_data, input_n1, input_n2, input_n3, input_n4, input_n5, input_n6, input_n7, input_n8, input_n9, input_n10, input_n11, input_n12, input_n13, input_n14, input_n15, input_n16, input_n17, input_n18, input_n19, input_n20, input_n21, input_n22, input_n23, input_n24, input_n25, input_n26, input_n27, input_n28, input_n29, input_n30, input_n31, input_n32, input_n33, input_n34, input_n35, input_n36, input_n37, input_n38, input_n39, input_n40, input_n41, input_n42, input_n43, input_n44, input_n45, input_n46, input_n47, input_n48, input_n49, region)
    
    pred_feature_maps_1 = feature_map_1, feature_map_2, feature_map_3
    pred_boxes_1, pred_confs_1, pred_probs_1 = yolo_model.predict(pred_feature_maps_1)
    pred_scores_1 = pred_confs_1 * pred_probs_1
    boxes_1, scores_1, labels_1 = gpu_nms(pred_boxes_1, pred_scores_1, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    pfm_2 = f1, f2, f3
    pb_2, pc_2, pp_2 = yolo_model.predict(pfm_2)
    ps_2 = pc_2 * pp_2
    b_2, s_2, l_2 = gpu_nms(pb_2, ps_2, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
    begin_fn = 3000
    end_fn = 5000
    distance = 15
    info = 0
    for i in range(video_frame_cnt):
        ret, img_ori = vid.read()
        if i in range(begin_fn, end_fn):
            if args.letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, tuple(args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            if i % distance == 0:
                _feature_map_1, _feature_map_2, _feature_map_3, _net1, _net2, _net3, _net4, _net5, _net6, _net7, _net8, _net9, _net10, _net11, _net12, _net13, _net14, _net15, _net16, _net17, _net18, _net19, _net20, _net21, _net22, _net23, _net24, _net25, _net26, _net27, _net28, _net29, _net30, _net31, _net32, _net33, _net34, _net35, _net36, _net37, _net38, _net39, _net40, _net41, _net42, _net43, _net44, _net45, _net46, _net47, _net48, _net49 = sess.run([feature_map_1, feature_map_2, feature_map_3, net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, net12, net13, net14, net15, net16, net17, net18, net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33, net34, net35, net36, net37, net38, net39, net40, net41, net42, net43, net44, net45, net46, net47, net48, net49], feed_dict = {input_data: img})
                boxes_, scores_, labels_ = sess.run([boxes_1, scores_1, labels_1], feed_dict={input_data: img})
                for j in range(1, 50):                   # store information of each layer
                    resfile = open('data/frame{}/l{}.txt'.format(i+1, j), 'w')
                    data = locals()['_net'+str(j)][0]
                    for data_slice in data:
                        np.savetxt(resfile, data_slice, fmt = '%.3f')
                    resfile.close()
                
                # rescale the coordinates to the original image
                if args.letterbox_resize:
                    boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                    boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
                else:
                    boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
                    boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))
                # save object detection results into 'od_res' file
                # done with full-inference frames
                res=np.arange(len(labels_) * 7).reshape(len(labels_), 7)
                res=res.astype(np.float32)
                res[:,0] = np.ones(len(labels_),np.float32) * (i + 1)
                res[:,1] = labels_
                res[:,2] = scores_.round(3)
                res[:,3:7] = boxes_.round(3)
                predres = open('data/od_res/res{}.txt'.format(i+1), 'w')
                np.savetxt(predres, res, fmt = '%.3f')
                print(res)
                predres.close()
                info = i
                prev_boxes = boxes_
            else:
                for j in range(1, 50):
                    locals()['data'+str(j)] = np.loadtxt('data/frame{}/l{}.txt'.format(info+1, j))
                    locals()['data'+str(j)] = np.asarray(locals()['data'+str(j)], np.float32)
                    locals()['data'+str(j)] = locals()['data'+str(j)].reshape(layersize[j-1], layersize[j-1], -1)
                    locals()['data'+str(j)] = locals()['data'+str(j)][np.newaxis, :]
                # calculate area
                RoIs = RoI_regularize(prev_boxes, i + 1)
                for RoI in RoIs:
                    xmin = RoI[0]
                    ymin = RoI[1]
                    h = RoI[3] - RoI[1] + 1
                    w = RoI[2] - RoI[0] + 1
                    ratio = 416 / 1920
                    r_w = int(ratio * w)
                    r_h = int(ratio * h)
                    xmin_2 = int(ratio * xmin)
                    ymin_2 = int(ratio * ymin + 91)
                    roi = [xmin_2, ymin_2, xmin_2 + r_w, ymin_2 + r_h]
                    img = img[:, roi[1]:roi[3], roi[0]:roi[2], :]
                    _b_2, _s_2, _l_2 = sess.run([b_2, s_2, l_2], feed_dict = {input_data: img, input_n1: data1, input_n2: data2, input_n3: data3, input_n4: data4, input_n5: data5, input_n6: data6, input_n7: data7, input_n8: data8, input_n9: data9, input_n10: data10, input_n11: data11, input_n12: data12, input_n13: data13, input_n14: data14, input_n15: data15, input_n16: data16, input_n17: data17, input_n18: data18, input_n19: data19, input_n20: data20, input_n21: data21, input_n22: data22, input_n23: data23, input_n24: data24, input_n25: data25, input_n26: data26, input_n27: data27, input_n28: data28, input_n29: data29, input_n30: data30, input_n31: data31, input_n32: data32, input_n33: data33, input_n34: data34, input_n35: data35, input_n36: data36, input_n37: data37, input_n38: data38, input_n39: data39, input_n40: data40, input_n41: dato41, input_n42: data42, input_n43: data43, input_n44: data44, input_n45: data45, input_n46: data46, input_n47: data47, input_n48: data48, input_n49: data49, region: roi})
                    if args.letterbox_resize:
                        _b_2[:, [0, 2]] = (_b_2[:, [0, 2]] - dw) / resize_ratio
                        _b_2[:, [1, 3]] = (_b_2[:, [1, 3]] - dh) / resize_ratio
                    else:
                        _b_2[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
                        _b_2[:, [1, 3]] *= (height_ori/float(args.new_size[1]))
                    res=np.arange(len(_l_2) * 7).reshape(len(_l_2), 7)
                    res=res.astype(np.float32)
                    res[:,0] = np.ones(len(_l_2),np.float32) * (i + 1)
                    res[:,1] = _l_2
                    res[:,2] = _s_2.round(3)
                    res[:,3:7] = _b_2.round(3)
                    predres = open('data/od_res/res{}.txt'.format(i+1), 'a')
                    np.savetxt(predres, res, fmt = '%.3f')
                    print(res)
                    predres.close()
            if args.save_video:
                videoWriter.write(img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        if args.save_video:
            videoWriter.release()
