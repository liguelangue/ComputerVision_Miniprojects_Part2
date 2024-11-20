# python WebCamSave.py -f ./test.mp4

import cv2 as cv
import numpy as np
import argparse
import time

# Initialize the parameters
confThreshold = 0.5  	# Confidence threshold
nmsThreshold = 0.4   	# Non-maximum suppression threshold
inpWidth = 416       	# Width of network's input image
inpHeight = 416      	# Height of network's input image

# Optical Flow Parameters
lk_params = dict(winSize=(15, 15),
                maxLevel=2,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Tracking state variables
tracking_points = None
prev_gray = None
mask = None
tracking_started = False
tracking_class_id = None  # Store the class ID of the tracked object
tracking_box = None  # Store the bounding box of the tracked object

# FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Set up argument parser
parser = argparse.ArgumentParser(description="Object Detection using YOLO with Optical Flow")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
args = parser.parse_args()

# Load names of classes and YOLO network
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
print('Using CPU device.')

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom, frame):
    # Draw detection box
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 8)
    label = '%.2f' % conf
    if classes:
        assert classId < len(classes)
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), 
                (left + round(1.5 * labelSize[0]), top + baseLine), 
                (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
    
    # Return center point of the detection
    center_x = left + (right - left) // 2
    center_y = top + (bottom - top) // 2
    return np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

def postprocess(frame, outs):
    global tracking_points, tracking_started, tracking_class_id, tracking_box
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []

    # First step: collect all detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            # Skip if the detected class is "person"
            if classes[classId] == "person":
                continue

            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    # Initialize tracking points for the first detected object
    if len(indices) > 0 and not tracking_started:
        i = indices[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        tracking_points = drawPred(classIds[i], confidences[i], left, top, 
                                 left + width, top + height, frame)
        tracking_started = True
        # Store the tracking object info
        tracking_class_id = classIds[i]
        tracking_box = box
    
    # Draw remaining detections
    for i in indices[1:]:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)

# Initialize video source
if args.file:
    print(f"Opening video file: {args.file}")
    vs = cv.VideoCapture(args.file)
else:
    print("Opening camera...")
    vs = cv.VideoCapture(0)
    time.sleep(2.0)

# Initialize video writer if specified
out = None
if args.out:
    frame_width = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vs.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(args.out, fourcc, fps, (frame_width, frame_height))

# Main loop
while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = int(1/(new_frame_time-prev_frame_time))
    prev_frame_time = new_frame_time

    # Convert current frame to grayscale
    current_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Initialize or update mask
    if mask is None:
        mask = np.zeros_like(frame)
    
    # Perform object detection
    if not tracking_started or tracking_points is None or len(tracking_points) < 1:
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)
    
    # Perform optical flow tracking
    if tracking_started and prev_gray is not None and tracking_points is not None:
        # Calculate optical flow
        new_points, status, error = cv.calcOpticalFlowPyrLK(prev_gray, current_gray, 
                                                          tracking_points, None, **lk_params)
        
        if new_points is not None:
            # Select good points
            good_new = new_points[status == 1]
            good_old = tracking_points[status == 1]

            # Draw tracking lines
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = map(int, [a, b, c, d])
                mask = cv.line(mask, (a, b), (c, d), (0, 255, 0), 6)
                frame = cv.circle(frame, (a, b), 12, (0, 0, 255), -1)

            tracking_points = good_new.reshape(-1, 1, 2)
            
            # Update box position based on point movement
            if len(good_new) > 0 and len(good_old) > 0:
                # Calculate average movement
                movement = np.mean(good_new - good_old, axis=0)
                tracking_box[0] += int(movement[0])  # Update left
                tracking_box[1] += int(movement[1])  # Update top
                # Draw the updated box
                left, top, width, height = tracking_box
                drawPred(tracking_class_id, 1.0, left, top, 
                        left + width, top + height, frame)
        else:
            tracking_started = False
            mask = np.zeros_like(frame)
    
    # Combine frame with tracking visualization
    output = cv.addWeighted(frame, 1, mask, 0.3, 0)
    
    # Draw FPS
    cv.putText(output, f'FPS: {fps}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
               1, (0, 0, 255), 2, cv.LINE_AA)

    # Write frame to output video if specified
    if out:
        out.write(output)

    # Display frame
    cv.imshow("Frame", output)
    
    # Handle key presses
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset tracking
        tracking_started = False
        tracking_points = None
        tracking_class_id = None
        tracking_box = None
        mask = np.zeros_like(frame)

    # Update previous frame
    prev_gray = current_gray.copy()

# Cleanup
vs.release()
if out:
    out.release()
cv.destroyAllWindows()