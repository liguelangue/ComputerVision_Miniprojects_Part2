import cv2 as cv
import numpy as np
import time

# Initialize the parameters
confThreshold = 0.5  	# Confidence threshold
nmsThreshold = 0.4   	# Non-maximum suppression threshold
inpWidth = 416       	# Width of network's input image
inpHeight = 416      	# Height of network's input image

# Initialize variables for recording
recording = False
vid_writer = None
output_file = ""
last_recording_time = 0  # Timestamp of last recording end
cooldown_time = 2  # Cooldown time in seconds

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
print('Using CPU device.')

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert classId < len(classes)
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), 
                 (left + round(1.5 * labelSize[0]), top + baseLine), 
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

# Remove the bounding boxes with low confidence
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    detected_classes = set()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
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
                # Add only "bottle" or "spoon" to detected_classes
                if classes[classId] in ["bottle", "spoon"]:
                    detected_classes.add(classes[classId])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    return detected_classes

# Start video capture from the webcam
cap = cv.VideoCapture(0)
fps = cap.get(cv.CAP_PROP_FPS) or 20.0  # Default to 20 FPS if unknown
target_frame_count = int(fps * 5)  # Frames needed for 5 seconds

winName = 'Object Detection with YOLO'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("No frame captured from camera. Exiting...")
        break

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    # Process detections
    detected_classes = postprocess(frame, outs)

    # Check if "bottle" or "spoon" is detected and start recording if cooldown has passed
    current_time = time.time()
    if detected_classes and not recording and (current_time - last_recording_time >= cooldown_time):
        detected_object = list(detected_classes)[0]  # Get detected object name
        recording = True
        frame_count = 0  # Reset frame count for new recording
        output_file = f"{detected_object}_group8.mp4"
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv.VideoWriter(output_file, fourcc, fps, (frame.shape[1], frame.shape[0]))
        print(f"Started recording {output_file}")

    # Continue recording if already started
    if recording:
        vid_writer.write(frame)
        frame_count += 1
        # Stop recording after reaching target frame count for 5 seconds
        if frame_count >= target_frame_count:
            recording = False
            last_recording_time = current_time  # Update the last recording end time
            vid_writer.release()
            print(f"Stopped recording {output_file}")

    # Display inference time
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Show the frame with detection boxes
    cv.imshow(winName, frame)

# Release resources
cap.release()
cv.destroyAllWindows()