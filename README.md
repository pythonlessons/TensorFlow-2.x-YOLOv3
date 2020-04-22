# TensorFlow-2.x-YOLOv3 tutorial

YOLOv3 implementation in TensorFlow 2.x, with support for training, transfer training.

## Installation
First, clode or download this GitHub repository.
Install requirements and download pretrained weights:
```
pip install -r ./requirements.txt
wget -P model_data https://pjreddie.com/media/files/yolov3.weights
```

## Quick start
Start with using pretrained weights to test predictions on both image and video:
```
python detection_demo.py
```

<p align="center">
    <img width="100%" src="IMAGES/city_pred.jpg" style="max-width:100%;">
    </a>
</p>

## Train custom dataset
To be continued...
