import cv2 
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the pre-trained model
model_path = 'vgg16_detector_V3.h5'
model = load_model(model_path)

# Define category labels
categories = ['Glasses', 'Phones', 'TV', 'Background']
category_colors = {"Glasses": (255, 0, 0), "Phones": (0, 255, 0), "TV": (0, 0, 255)}

# Start the camera
cap = cv2.VideoCapture(0)

# Initialize selective search and tracker list
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
trackers = []  # List to store multiple trackers and category information

detect_mode = False  # Initial state is not detecting

# FPS calculation initialization
fps = 0
prev_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the image to improve selective search efficiency
    resized_frame = cv2.resize(frame, (256, 256))
    orig_height, orig_width = frame.shape[:2]

    # Detection mode
    if detect_mode:
        ss.setBaseImage(resized_frame)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()[:50]  # Limit the number of proposals to 50

        # Clear old trackers
        trackers = []

        # Iterate through region proposals and identify high-confidence targets
        for result in ssresults:
            x, y, w, h = result
            roi = resized_frame[y:y + h, x:x + w]
            resized_roi = cv2.resize(roi, (224, 224))
            resized_roi = np.expand_dims(resized_roi, axis=0)

            # Perform category prediction
            predictions = model.predict(resized_roi)
            class_id = np.argmax(predictions)
            score = np.max(predictions)

            if score > 0.8 and class_id < len(categories) - 1:  # Exclude background
                class_name = categories[class_id]
                color = category_colors[class_name]

                # Calculate original coordinates and initialize tracker
                orig_x1 = int(x * (orig_width / 256))
                orig_y1 = int(y * (orig_height / 256))
                orig_x2 = int((x + w) * (orig_width / 256))
                orig_y2 = int((y + h) * (orig_height / 256))

                # Draw detected bounding box and category label
                cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 2)
                cv2.putText(frame, f"{class_name}: {score:.2f}", (orig_x1, orig_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Initialize tracker and pass bounding box
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1))
                
                # Store tracker and category information in the list
                trackers.append((tracker, class_name))

        detect_mode = False  # Turn off detection mode

    # Tracking mode
    for tracker, tracked_class_name in trackers:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            color = category_colors.get(tracked_class_name, (0, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{tracked_class_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS
    current_time = cv2.getTickCount()
    time_elapsed = (current_time - prev_time) / cv2.getTickFrequency()
    prev_time = current_time
    fps = 1 / time_elapsed if time_elapsed > 0 else 0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display image
    cv2.imshow('Object Detection and Tracking', frame)

    # Key control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('d'):  # Press 'd' to start detection
        detect_mode = True
        trackers = []  # Clear old trackers
    elif key == ord('r'):  # Press 'r' to re-detect
        detect_mode = True
        trackers = []  # Clear old trackers

# Release resources
cap.release()
cv2.destroyAllWindows()