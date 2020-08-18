#================================================================
#
#   File name   : Convert_to_TRT.py
#   Author      : PyLessons
#   Created date: 2020-08-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : convert TF frozen graph to TensorRT model
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
import numpy as np
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from yolov3.configs import *
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def calibration_input():
    for i in range(10):
        batched_input = np.random.random((1, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3)).astype(np.float32)
        batched_input = tf.constant(batched_input)
        yield (batched_input,)

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=4000000000)
conversion_params = conversion_params._replace(precision_mode=YOLO_TRT_QUANTIZE_MODE)
conversion_params = conversion_params._replace(max_batch_size=1)
if YOLO_TRT_QUANTIZE_MODE == 'INT8':
    conversion_params = conversion_params._replace(use_calibration=True)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=f'./checkpoints/{YOLO_TYPE}-{YOLO_INPUT_SIZE}', conversion_params=conversion_params)
if YOLO_TRT_QUANTIZE_MODE == 'INT8':
    converter.convert(calibration_input_fn=calibration_input)
else:
    converter.convert()

converter.save(output_saved_model_dir=f'./checkpoints/{YOLO_TYPE}-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}')
print(f'Done Converting to TensorRT, model saved to: /checkpoints/{YOLO_TYPE}-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}')
