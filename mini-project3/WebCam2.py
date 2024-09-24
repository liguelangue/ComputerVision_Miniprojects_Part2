import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is captured
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Stack the original and gray images horizontally
    combined = cv2.hconcat([frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])

    # Display the resulting frame
    cv2.imshow('Original and Grayscale', combined)

    # Get the height and width of the frame
    h, w = frame.shape[:2]

    # Translation - shift the img horizontally and vertically
    tx, ty = 50, 30
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_frame = cv2.warpAffine(frame, translation_matrix, (w, h))

    # Rotation
    center = (w // w, h // 2)
    angle = 45
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(frame, rotation_matrix, (h, w))

    # Scaling 
    scale_size = 0.5
    scaled_img = cv2.resize(frame, None, fx = scale_size, fy = scale_size)

    # Perspective Transformation
    p1 = np.float32([[0,0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    p2 = np.float32([[0,0], [w - 1, 0], [w * 0.2, h - 1], [w * 0.8, h - 1]])
    perspective_matrix = cv2.getPerspectiveTransform(p1, p2)
    perspective_frame = cv2.warpPerspective(frame, perspective_matrix, (w, h))


    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('t'):
        cv2.imshow('translation', translated_frame)
    if key == ord('r'):
        cv2.imshow('rotated', rotated_img)
    if key == ord('s'):
        cv2.imshow('scaled', scaled_img)
    if key == ord('p'):
        cv2.imshow('perspective', perspective_frame)
    if key == ord('q') & 0xFF:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
