import cv2
import numpy as np

def create_blank_image(height, width):
    return np.zeros((height, width, 3), dtype=np.uint8)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

# Initialize the mode flags
translation_on = False
rotation_on = False
scaling_on = False
perspective_on = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is captured
    if not ret:
        break
    
    # Get the height and width of the frame
    h, w = frame.shape[:2]
    
    # Create a copy of the frame to apply transformations
    processed = frame.copy()
    
    # Apply transformations based on flags
    if translation_on:
        tx, ty = 500, 300
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        processed = cv2.warpAffine(processed, translation_matrix, (w, h))

    if rotation_on:
        center = (w // 2, h // 2)
        angle = 45
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        processed = cv2.warpAffine(processed, rotation_matrix, (w, h))

    if scaling_on:
        scale_size = 0.75
        processed = cv2.resize(processed, None, fx=scale_size, fy=scale_size)
        processed = cv2.resize(processed, (w, h))

    if perspective_on:
        p1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
        p2 = np.float32([[0,0], [w-1,0], [w*0.2,h-1], [w*0.8,h-1]])
        perspective_matrix = cv2.getPerspectiveTransform(p1, p2)
        processed = cv2.warpPerspective(processed, perspective_matrix, (w, h))
    
    # Combine original and processed images side by side
    combined = cv2.hconcat([frame, processed])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Processed', (w+10, 30), font, 1, (255, 255, 255), 2)
    
    # Add status indicators
    status_y = h - 10
    if translation_on:
        cv2.putText(combined, 'T', (w+10, status_y), font, 0.7, (0, 255, 0), 2)
    if rotation_on:
        cv2.putText(combined, 'R', (w+40, status_y), font, 0.7, (0, 255, 0), 2)
    if scaling_on:
        cv2.putText(combined, 'S', (w+70, status_y), font, 0.7, (0, 255, 0), 2)
    if perspective_on:
        cv2.putText(combined, 'P', (w+100, status_y), font, 0.7, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Original and Processed', combined)
    
    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    
    # Toggle transformations based on key press
    if key == ord('t'):
        translation_on = not translation_on
    elif key == ord('r'):
        rotation_on = not rotation_on
    elif key == ord('s'):
        scaling_on = not scaling_on
    elif key == ord('p'):
        perspective_on = not perspective_on
    elif key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()