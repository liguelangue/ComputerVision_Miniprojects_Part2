# CS5330_F24_Group8_Mini_Project_11

## Project Members

Anning Tian, Pingyi Xu, Qinhao Zhang, Xinmeng Wu

## Setup Instructions

Put these files in the same folder with the two python files:

coco.names

yolov3.cfg

yolov3.weights

test.mp4

test_multi.mp4

The link to these files: https://drive.google.com/drive/folders/1XJVkmjs07MyUfxbpWrGK2UBibHBCpomo?usp=sharing

## Usage Guide

To use the camera:

``` python WebCamSave.py ```

``` python WebCamSave_multi.py ```

To use the test video:

``` python WebCamSave.py -f ./test.mp4```

``` python WebCamSave_multi.py -f ./test_multi.mp4```

'r': Reset the object detection and tracking

'q': Exit the application

## Description of the Project

The target objects for tracking:

79 classes in COCO dataset. ('Person' is not included for better video demonstration)

The YOLO model used:

YOLOv3

The optical flow algorithm implemented:

Lucas-Kanade

## Video Demonstration
The link for the demonstration:

https://drive.google.com/file/d/1EX7cbYdBbuBZkX7KqaXtBF__6fG-SMxg/view?usp=sharing

The link for the test videos:

test.mp4 https://drive.google.com/file/d/1QFZKJCFow1sgOgm7_HgE0Vur0okpzxZk/view?usp=sharing

test_multi.mp4 https://drive.google.com/file/d/16FzDbTcRco2iUYE1zBAd9jYKx4tEdu0d/view?usp=sharing