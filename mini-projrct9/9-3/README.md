# Target Classes
bottle

spoon

# Setup Instructions
When running this script, please ensure that these files are located in the same directory:

coco.names

yolov3.cfg

yolov3.weights

# Usage Guide
```
python WebCamSave_Yolo.py
```
When the camera is on, all classes in the COCO dataset will be tracked. When the target classes are detected, a 5-second video will be saved. Once recording starts, the message 'Started recording' will be displayed. Please wait until 'Stopped recording' appears to ensure a complete 5-second video file.