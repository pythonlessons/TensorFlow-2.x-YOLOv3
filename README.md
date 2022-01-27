# TensorFlow-2.x-YOLOv3 and YOLOv4 test

I tested YOLOv3 and YOLOv4 with help of this original implementation from [pythonlessons](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3)
If you want to read original README, please read [this](README.orig.md)
Original tested environment is as follows:

- OS Ubuntu 18.04
- CUDA 10.1
- cuDNN v7.6.5
- TensorRT-6.0.1.5
- Tensorflow-GPU 2.3.1
- Code was tested on Ubuntu and Windows 10 (TensorRT not supported officially)


However, I strongly recommend using Docker because you can prepare environment very easily. 
So I use the following command with OS Ubuntu 20.04 and NVIDIA Titan-X GPU.

Before the following, check the NVIDIA driver status with `nvidia-smi`.

```bash
$ docker run --gpus all -it --rm -v ~/Workspace:/Workspace nvcr.io/nvidia/tensorrt:19.10-py3
```



## Prepare
First, clone or download this GitHub repository.
Install requirements.
```
pip install -r ./requirements.txt

# if you don't install the following lib, libSM.so.6 error occurs

apt update
apt install libsm6 libxext6 libxrender-dev
```

Download pretrained weights:
```
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov4
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

If you don't need to test tiny models, you don't need to download the following weights
```
# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights

# yolov4-tiny
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

## Quick start
Start with using pretrained weights to test predictions on both image and video:
```
python detection_demo.py
```

## Converting YOLO to TensorRT
I will give two examples, both will be for YOLOv4 model,quantize_mode=INT8 and model input size will be 608. Detailed tutorial is on this [link](https://pylessons.com/YOLOv4-TF2-TensorRT/).
### Default weights from COCO dataset:
- Download weights from links above;
- In `configs.py` script choose your `YOLO_TYPE`;
- In `configs.py` script set `YOLO_INPUT_SIZE = 608`;
- In `configs.py` script set `YOLO_FRAMEWORK = "trt"`;
- From main directory in terminal type `python tools/Convert_to_pb.py`;
  - It will save frozen model of `YOLO_TYPE`(ex. yolov4-608) to `checkpoints` path
- From main directory in terminal type `python tools/Convert_to_TRT.py`;
  - It will take a way too long time, so you should be patient
  - With my dev environment, it took around **30 minutes** with YOLOv4
- In `configs.py` script set `YOLO_CUSTOM_WEIGHTS = f'checkpoints/{YOLO_TYPE}-trt-{YOLO_TRT_QUANTIZE_MODE}–{YOLO_INPUT_SIZE}'`;
- Now you can run `detection_demo.py`, best to test with `detect_video` function.
