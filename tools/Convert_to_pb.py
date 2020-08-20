#================================================================
#
#   File name   : Convert_to_pb.py
#   Author      : PyLessons
#   Created date: 2020-08-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to freeze tf model to .pb model
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

foldername = os.path.basename(os.getcwd())
if foldername == "tools":
    os.chdir("..")
sys.path.insert(1, os.getcwd())

import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights
from yolov3.configs import *

if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
if YOLO_CUSTOM_WEIGHTS == False:
    load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
else:
    yolo.load_weights(YOLO_CUSTOM_WEIGHTS) # use custom weights

yolo.summary()
yolo.save(f'./checkpoints/{YOLO_TYPE}-{YOLO_INPUT_SIZE}')

print(f"model saves to /checkpoints/{YOLO_TYPE}-{YOLO_INPUT_SIZE}")
