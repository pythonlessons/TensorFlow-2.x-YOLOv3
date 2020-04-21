#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-04-20
#   Website     : https://pylessons.com/
#   GitHub      :
#   Description : object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_image, detect_video
from yolov3.configs import *

input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_DARKNET_WEIGHTS

image_path   = "./IMAGES/kite.jpg"
video_path   = "./IMAGES/street_drive.mp4"

yolo = Create_Yolov3(input_size=input_size)
load_yolo_weights(yolo, Darknet_weights) # use Darknet weights

detect_image(yolo, image_path, '', input_size=input_size, show=True, rectangle_colors=(255,0,0))
#detect_video(yolo, video_path, '', input_size=input_size, show=True, rectangle_colors=(255,0,0))
