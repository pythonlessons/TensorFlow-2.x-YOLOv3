#================================================================
#
#   File name   : detect_mnist.py
#   Author      : PyLessons
#   Created date: 2020-04-20
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : mnist object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import detect_image
from yolov3.configs import *

input_size=YOLO_INPUT_SIZE

while True:
    ID = random.randint(0, 200)
    label_txt = "mnist/mnist_test.txt"
    image_info = open(label_txt).readlines()[ID].split()

    image_path = image_info[0]

    yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
    yolo.load_weights("./checkpoints/yolov3_custom_Tiny") # use keras weights

    detect_image(yolo, image_path, "mnist_test.jpg", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    time.sleep(5)

