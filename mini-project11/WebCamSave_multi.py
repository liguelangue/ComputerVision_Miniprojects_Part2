# python WebCamSave_multi.py -f ./test_multi.mp4

import cv2 as cv
import numpy as np
import argparse
import time

# Initialize the parameters
confThreshold = 0.5  	# Confidence threshold
nmsThreshold = 0.4 	# Non-maximum suppression threshold
inpWidth = 416       	# Width of network's input image
inpHeight = 416      	# Height of network's input image

# Optical Flow Parameters
lk_params = dict(winSize=(15, 15),
                maxLevel=2,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Add a class to store tracking object information
class TrackedObject:
    def __init__(self, points, class_id, box, confidence):
        self.points = points  # Tracking points
        self.class_id = class_id  # Class ID
        self.box = list(box)  # Bounding box [left, top, width, height]
        self.confidence = confidence  # Confidence
        self.color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for each object

# Global variables
tracked_objects = []  # List to store multiple tracking objects
tracking_started = False
prev_gray = None
mask = None

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
    global tracked_objects, tracking_started
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []

    # Step 1: Collect all detections
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

    # Step 2: Perform NMS by class
    final_indices = []
    # Group by class
    unique_classes = set(classIds)
    
    for cls in unique_classes:
        # Get all boxes for this class
        class_indices = [i for i, cid in enumerate(classIds) if cid == cls]
        if not class_indices:
            continue
            
        # Get boxes and confidences for this class
        class_boxes = [boxes[i] for i in class_indices]
        class_confidences = [confidences[i] for i in class_indices]
        
        # Perform NMS for this class
        nms_indices = cv.dnn.NMSBoxes(class_boxes, class_confidences, 
                                     confThreshold, nmsThreshold)
        
        # Add NMS results for this class to final results
        final_indices.extend([class_indices[i] for i in nms_indices])

    # Initialize tracking when objects are detected
    if len(final_indices) > 0:
        # If tracking hasn't started, clear old tracking objects
        if not tracking_started:
            tracked_objects.clear()
            tracking_started = True
        
        # Process all detected objects
        for i in final_indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            # Calculate IoU with existing tracked objects
            add_new_object = True
            if tracked_objects:
                # Calculate IoU between current box and all tracked objects
                current_box = [left, top, width, height]
                for tracked_obj in tracked_objects:
                    iou = calculate_iou(current_box, tracked_obj.box)
                    if iou > 0.1:  # IoU threshold
                        add_new_object = False
                        break
            
            if add_new_object:
                center_point = np.array([[left + width//2, top + height//2]], 
                                      dtype=np.float32).reshape(-1, 1, 2)
                
                # Create new tracking object
                tracked_obj = TrackedObject(
                    points=center_point,
                    class_id=classIds[i],
                    box=box,
                    confidence=confidences[i]
                )
                tracked_objects.append(tracked_obj)
            
            # Draw detection box and label
            drawPred(classIds[i], confidences[i], left, top, 
                    left + width, top + height, frame)

# Add IoU calculation function
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    # Convert to [x1, y1, x2, y2] format
    b1_x1, b1_y1 = box1[0], box1[1]
    b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1 = box2[0], box2[1]
    b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection area
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Calculate intersection area
    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * \
                 max(0, inter_rect_y2 - inter_rect_y1)

    # Calculate area of both boxes
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # Calculate IoU
    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou

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
    
    # Always perform object detection
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)
    
    # Perform optical flow tracking for each object
    if tracking_started and prev_gray is not None:
        new_tracked_objects = []
        
        for tracked_obj in tracked_objects:
            if tracked_obj.points is not None:
                # Calculate optical flow
                new_points, status, error = cv.calcOpticalFlowPyrLK(
                    prev_gray, current_gray, tracked_obj.points, None, **lk_params)
                
                if new_points is not None:
                    # Select good points
                    good_new = new_points[status == 1]
                    good_old = tracked_obj.points[status == 1]

                    # Draw tracking lines with object-specific color
                    for new, old in zip(good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        a, b, c, d = map(int, [a, b, c, d])
                        mask = cv.line(mask, (a, b), (c, d), tracked_obj.color, 6)  # Thick line
                        frame = cv.circle(frame, (a, b), 12, tracked_obj.color, -1)  # Large point

                    # Update tracking points
                    tracked_obj.points = good_new.reshape(-1, 1, 2)
                    
                    # Update bounding box position
                    if len(good_new) > 0 and len(good_old) > 0:
                        mean_movement = np.mean(good_new - good_old, axis=0)
                        tracked_obj.box[0] += int(mean_movement[0])  # Update left
                        tracked_obj.box[1] += int(mean_movement[1])  # Update top
                        
                        # Draw updated bounding box and label
                        box = tracked_obj.box
                        drawPred(tracked_obj.class_id, tracked_obj.confidence,
                               box[0], box[1], box[0] + box[2], box[1] + box[3], frame)
                    
                    new_tracked_objects.append(tracked_obj)
        
        tracked_objects = new_tracked_objects
        
        if not tracked_objects:  # If all objects lost tracking
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
        tracked_objects = []
        mask = np.zeros_like(frame)

    # Update previous frame
    prev_gray = current_gray.copy()

# Cleanup
vs.release()
if out:
    out.release()
cv.destroyAllWindows()