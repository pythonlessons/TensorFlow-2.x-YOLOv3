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
  - With my dev environment, it took around **55 minutes** with YOLOv4
  ```
  Done Converting to TensorRT, model saved to: /checkpoints/yolov4-trt-INT8-608
  ```

  During conversion, you can check nvidia-smi that there is a process 'python' running
```
$ nvidia-smi
Thu Jan 27 16:18:14 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN X ...  Off  | 00000000:19:00.0 Off |                  N/A |
| 27%   44C    P8    17W / 250W |   9637MiB / 12192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    772207      C   python                           9635MiB |
+-----------------------------------------------------------------------------+
```
- In `configs.py` script set `YOLO_CUSTOM_WEIGHTS = f'checkpoints/{YOLO_TYPE}-trt-{YOLO_TRT_QUANTIZE_MODE}–{YOLO_INPUT_SIZE}'`;
- Now you can run `detection_demo.py`, best to test with `detect_video` function.

```
root@971f9b73b337:/Workspace/TensorFlow-2.x-YOLOv3# python detection_demo.py
2022-01-27 07:22:33.278558: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2022-01-27 07:22:34.392922: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2022-01-27 07:22:34.922515: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:
pciBusID: 0000:19:00.0 name: NVIDIA TITAN X (Pascal) computeCapability: 6.1
coreClock: 1.531GHz coreCount: 28 deviceMemorySize: 11.91GiB deviceMemoryBandwidth: 447.48GiB/s
2022-01-27 07:22:34.922585: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2022-01-27 07:22:34.925841: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2022-01-27 07:22:34.928669: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2022-01-27 07:22:34.929128: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2022-01-27 07:22:34.931549: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2022-01-27 07:22:34.932925: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2022-01-27 07:22:34.937907: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2022-01-27 07:22:34.938971: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
GPUs [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
2022-01-27 07:25:06.747241: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-01-27 07:25:06.770352: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3699850000 Hz
2022-01-27 07:25:06.771865: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x29d02e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-01-27 07:25:06.771902: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-01-27 07:25:06.864253: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x49b0420 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2022-01-27 07:25:06.864303: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA TITAN X (Pascal), Compute Capability 6.1
2022-01-27 07:25:06.865643: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:
pciBusID: 0000:19:00.0 name: NVIDIA TITAN X (Pascal) computeCapability: 6.1
coreClock: 1.531GHz coreCount: 28 deviceMemorySize: 11.91GiB deviceMemoryBandwidth: 447.48GiB/s
2022-01-27 07:25:06.865706: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2022-01-27 07:25:06.865754: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2022-01-27 07:25:06.865780: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2022-01-27 07:25:06.865806: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2022-01-27 07:25:06.865835: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2022-01-27 07:25:06.865858: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2022-01-27 07:25:06.865882: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2022-01-27 07:25:06.867937: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2022-01-27 07:25:06.868000: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2022-01-27 07:25:07.153676: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-01-27 07:25:07.153716: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0
2022-01-27 07:25:07.153720: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N
2022-01-27 07:25:07.154663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11227 MB memory) -> physical GPU (device: 0, name: NVIDIA TITAN X (Pascal), pci bus id: 0000:19:00.0, compute capability: 6.1)
2022-01-27 07:28:36.814630: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libnvinfer.so.6
[ERROR:0] global /io/opencv/modules/videoio/src/cap.cpp (392) open VIDEOIO(CV_IMAGES): raised OpenCV exception:
OpenCV(4.1.2) /io/opencv/modules/videoio/src/cap_images.cpp:207: error: (-215:Assertion failed) !filename.empty() in function 'icvExtractPattern'

Time: 2031.85ms, Detection FPS: 0.5, total FPS: 0.5
Time: 1031.46ms, Detection FPS: 1.0, total FPS: 0.9
Time: 697.91ms, Detection FPS: 1.4, total FPS: 1.4
Time: 531.18ms, Detection FPS: 1.9, total FPS: 1.8
Time: 431.15ms, Detection FPS: 2.3, total FPS: 2.2
Time: 364.41ms, Detection FPS: 2.7, total FPS: 2.6
Time: 316.54ms, Detection FPS: 3.2, total FPS: 2.9
Time: 280.60ms, Detection FPS: 3.6, total FPS: 3.3
Time: 252.73ms, Detection FPS: 4.0, total FPS: 3.6
Time: 230.44ms, Detection FPS: 4.3, total FPS: 3.9
Time: 212.11ms, Detection FPS: 4.7, total FPS: 4.2
Time: 196.87ms, Detection FPS: 5.1, total FPS: 4.5
Time: 183.93ms, Detection FPS: 5.4, total FPS: 4.8
Time: 172.80ms, Detection FPS: 5.8, total FPS: 5.1
Time: 163.19ms, Detection FPS: 6.1, total FPS: 5.3
Time: 154.74ms, Detection FPS: 6.5, total FPS: 5.6
Time: 147.32ms, Detection FPS: 6.8, total FPS: 5.8
Time: 140.72ms, Detection FPS: 7.1, total FPS: 6.1
Time: 134.77ms, Detection FPS: 7.4, total FPS: 6.3
Time: 129.46ms, Detection FPS: 7.7, total FPS: 6.5
Time: 29.30ms, Detection FPS: 34.1, total FPS: 18.9
Time: 29.07ms, Detection FPS: 34.4, total FPS: 19.0
Time: 28.85ms, Detection FPS: 34.7, total FPS: 19.2
Time: 28.57ms, Detection FPS: 35.0, total FPS: 19.4
Time: 28.35ms, Detection FPS: 35.3, total FPS: 19.4
Time: 28.11ms, Detection FPS: 35.6, total FPS: 19.6
Time: 27.93ms, Detection FPS: 35.8, total FPS: 19.7
Time: 27.80ms, Detection FPS: 36.0, total FPS: 19.8
Time: 27.63ms, Detection FPS: 36.2, total FPS: 19.9
Time: 27.46ms, Detection FPS: 36.4, total FPS: 20.0
Time: 27.32ms, Detection FPS: 36.6, total FPS: 20.1
Time: 27.14ms, Detection FPS: 36.8, total FPS: 20.3
Time: 27.00ms, Detection FPS: 37.0, total FPS: 20.4
Time: 26.90ms, Detection FPS: 37.2, total FPS: 20.4
Time: 26.79ms, Detection FPS: 37.3, total FPS: 20.4
Time: 26.70ms, Detection FPS: 37.4, total FPS: 20.4
Time: 26.60ms, Detection FPS: 37.6, total FPS: 20.4
Time: 26.47ms, Detection FPS: 37.8, total FPS: 20.6
Time: 26.37ms, Detection FPS: 37.9, total FPS: 20.6
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.6
Time: 26.10ms, Detection FPS: 38.3, total FPS: 20.7
Time: 26.09ms, Detection FPS: 38.3, total FPS: 20.7
Time: 26.11ms, Detection FPS: 38.3, total FPS: 20.6
Time: 26.19ms, Detection FPS: 38.2, total FPS: 20.4
Time: 26.17ms, Detection FPS: 38.2, total FPS: 20.5
Time: 26.20ms, Detection FPS: 38.2, total FPS: 20.5
Time: 26.17ms, Detection FPS: 38.2, total FPS: 20.7
Time: 26.15ms, Detection FPS: 38.2, total FPS: 20.7
Time: 26.16ms, Detection FPS: 38.2, total FPS: 20.7
Time: 26.17ms, Detection FPS: 38.2, total FPS: 20.7
Time: 26.18ms, Detection FPS: 38.2, total FPS: 20.7
Time: 26.22ms, Detection FPS: 38.1, total FPS: 20.7
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.6
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.7
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.4
Time: 26.22ms, Detection FPS: 38.1, total FPS: 20.4
Time: 26.22ms, Detection FPS: 38.1, total FPS: 20.3
Time: 26.26ms, Detection FPS: 38.1, total FPS: 20.2
Time: 26.26ms, Detection FPS: 38.1, total FPS: 20.3
Time: 26.28ms, Detection FPS: 38.0, total FPS: 20.2
Time: 26.31ms, Detection FPS: 38.0, total FPS: 20.2
Time: 26.33ms, Detection FPS: 38.0, total FPS: 20.2
Time: 26.34ms, Detection FPS: 38.0, total FPS: 20.3
Time: 26.31ms, Detection FPS: 38.0, total FPS: 20.3
Time: 26.33ms, Detection FPS: 38.0, total FPS: 20.3
Time: 26.32ms, Detection FPS: 38.0, total FPS: 20.3
Time: 26.38ms, Detection FPS: 37.9, total FPS: 20.1
Time: 26.46ms, Detection FPS: 37.8, total FPS: 20.0
Time: 26.42ms, Detection FPS: 37.9, total FPS: 20.1
Time: 26.35ms, Detection FPS: 38.0, total FPS: 20.3
Time: 26.32ms, Detection FPS: 38.0, total FPS: 20.3
Time: 26.30ms, Detection FPS: 38.0, total FPS: 20.2
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.4
Time: 26.22ms, Detection FPS: 38.1, total FPS: 20.4
Time: 26.19ms, Detection FPS: 38.2, total FPS: 20.8
Time: 26.19ms, Detection FPS: 38.2, total FPS: 20.8
Time: 26.18ms, Detection FPS: 38.2, total FPS: 21.0
Time: 26.17ms, Detection FPS: 38.2, total FPS: 21.0
Time: 26.21ms, Detection FPS: 38.2, total FPS: 20.8
Time: 26.21ms, Detection FPS: 38.1, total FPS: 20.8
Time: 26.20ms, Detection FPS: 38.2, total FPS: 20.8
Time: 26.15ms, Detection FPS: 38.2, total FPS: 20.9
Time: 26.11ms, Detection FPS: 38.3, total FPS: 21.0
Time: 26.12ms, Detection FPS: 38.3, total FPS: 21.0
Time: 26.11ms, Detection FPS: 38.3, total FPS: 21.0
Time: 26.10ms, Detection FPS: 38.3, total FPS: 21.0
Time: 26.10ms, Detection FPS: 38.3, total FPS: 21.0
Time: 26.06ms, Detection FPS: 38.4, total FPS: 21.1
Time: 26.09ms, Detection FPS: 38.3, total FPS: 21.0
Time: 26.14ms, Detection FPS: 38.3, total FPS: 20.9
Time: 26.15ms, Detection FPS: 38.2, total FPS: 20.8
Time: 26.19ms, Detection FPS: 38.2, total FPS: 20.8
Time: 26.21ms, Detection FPS: 38.1, total FPS: 20.7
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.7
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.6
Time: 26.26ms, Detection FPS: 38.1, total FPS: 20.5
Time: 26.28ms, Detection FPS: 38.1, total FPS: 20.5
Time: 26.24ms, Detection FPS: 38.1, total FPS: 20.6
Time: 26.25ms, Detection FPS: 38.1, total FPS: 20.5
```