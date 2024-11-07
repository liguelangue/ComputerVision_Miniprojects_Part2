import cv2
import numpy as np
import joblib
from keras.models import load_model
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Load the trained model
model = load_model("saved_model/model_v1.h5")

# Load the label binarizer
lb = joblib.load("saved_model/label_binarizer.pkl")

# Initialize webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Define preprocessing function (update to match your model's input requirements)
def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))  # Resize frame to match model input
    frame = frame.astype("float32") / 255.0  # Normalize the image and convert to float32
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

current_label = "detecting"
previous_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Predict class probabilities
    predictions = model.predict(input_frame, verbose=0)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    # Get class label if confidence is above 60%
    if confidence >= 0.6:
        current_label = lb.classes_[class_index]
    else:
        current_label = "detecting"

    # Overlay the label and FPS on the frame
    overlay_text = f"{current_label}"
    cv2.putText(frame, overlay_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Webcam Feed', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Break loop on 'q' key press
    if key == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
