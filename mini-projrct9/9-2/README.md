# CS5330_F24_Group8_Mini_Project_9-2

## Project Members
Anning Tian, Pingyi Xu, Qinhao Zhang, Xinmeng Wu

## Overview
This project implements real-time object detection and tracking using a pre-trained VGG16 model, with selective search for region proposals and KCF tracker for tracking detected objects. It uses OpenCV for image processing and TensorFlow for deep learning. The program detects and tracks objects (e.g., glasses, phones, TVs) in a live webcam feed and displays the FPS (frames per second) and detection results in real time.

## Setup Instructions
Download the zip file or use GitHub Desktop to clone the file folder.


WebCamSave_V3.py: This file is used to apply the trained model in a live setting using a webcam. It captures live video streams and uses the trained model to make predictions on the captured frames, demonstrating the real-time performance of the model.

The saved_model folder contains the following files:
- vgg16_detector_V3.h5: This is the trained model file, which stores the network weights and structure to be used for making predictions on new data.

## usage guide
Run the LiveCam file:
```
python WebCamSave_V3.py
```

In the upper left corner, it shows the FPS, and after detection, different objects are highlighted in different colors.

# Dataset Information and Catalog：
- Link: https://drive.google.com/drive/folders/1lepJM4a6fBqTWQajXSQDB5o8iK8lRQn8?usp=sharing
- Catalog: Phone, TV, Glass


## Details about the dataset
Phone: 584

Glass: 139

TV: 386

## Data Preprocessing
### Resize Frame：
The current frame is resized to improve selective search efficiency.
### Selective Search for Region Proposals:
Generates 50 region proposals for object detection

# Detection Process
## Object Classfication and High Conidence Filtering:
For each proposal, a region of interest(ROI) is extracted and passed through the model to determine the calss and confidence score. Only objects with a confidence score > 0.8 are displayed excluding background.
## KCF Tracker Initialization:
A KCF tracker is created for each detected object to track it across frames.

## Track Process:
In tracking mode, the program updates the position of each tracker and redraws bounding boxes for each object, along with its category label. The bounding box color corresponds to the category.(Glassws - Red, TVs - Blue, Phones - Green)

# Display and Control:
- FPS is shown in the upper left corner.
- if press "D" or "d", the detect model will start
- if press "R" or "r", the detect will restart and clear the old tracker.
- if press "q" or "Q", it will quit.


## Optimized Parameters
- Confidence Threshold: Only objects with a confidence score greater than 0.8 are displayed to avoid false positives.
- Region Proposals: Limited to the top 50 proposals in selective search to reduce processing time.
- Frame Resize: Each frame is resized to 256x256 pixels to enhance selective search and model processing speed.
- Tracking Algorithm: Uses KCF Tracker as it balances speed and accuracy for real-time tracking.

## The link to the video demonstrating
