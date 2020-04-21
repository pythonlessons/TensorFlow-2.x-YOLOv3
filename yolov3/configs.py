#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   File name   : configs.py
#   Author      : PyLessons
#   Created date: 2020-04-20
#   Website     : https://pylessons.com/
#   GitHub      :
#   Description : yolov3 configuration file
#
#================================================================

# YOLO options
YOLO_DARKNET_WEIGHTS        = "./model_data/yolov3.weights"
YOLO_COCO_CLASSES           = "./model_data/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
YOLO_ANCHORS                = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]

