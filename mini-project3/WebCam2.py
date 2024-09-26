import cv2
import numpy as np
import time

def create_blank_image(height, width):
    return np.zeros((height, width, 3), dtype=np.uint8)

def draw_status_indicator(image, text, info, is_active, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text + info, font, 0.7, 2)
    text_w, text_h = text_size
    x, y = position
    
    padding = 20  # Increase padding for wider rectangles
    rect_w = text_w + 4 * padding  # Increase horizontal size more significantly
    rect_h = text_h + 2 * padding  # Keep vertical size the same
    
    x = x - rect_w // 2

    if is_active:
        # Green background with transparent text
        cv2.rectangle(image, (x, y), (x + rect_w, y + rect_h), (0, 255, 0), -1)
        cv2.putText(image, text + info, (x + padding, y + rect_h - padding), font, 0.7, (0, 0, 0), 2)
    else:
        # Transparent background with solid text
        cv2.rectangle(image, (x, y), (x + rect_w, y + rect_h), (255, 255, 255), -1)
        cv2.putText(image, text + info, (x + padding, y + rect_h - padding), font, 0.7, (0, 0, 0), 2)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the mode flags
translation_on = False
rotation_on = False
scaling_on = False
perspective_on = False

# Initialize variables
scale = 1.0
prev_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Get the height and width of the frame
    h, w = frame.shape[:2]
    
    # Create a copy of the frame to apply transformations
    processed = frame.copy()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if scaling_on:
        center_x, center_y = w // 2, h // 2
        new_w = int(w / scale)
        new_h = int(h / scale)
        if new_w < w and new_h < h:
            start_x = max(0, center_x - new_w // 2)
            start_y = max(0, center_y - new_h // 2)
            end_x = min(w, center_x + new_w // 2)
            end_y = min(h, center_y + new_h // 2)
            cropped_frame = frame[start_y:end_y, start_x:end_x]
            processed = cv2.resize(cropped_frame, (w, h))
        
        if key == ord('=') or key == ord('+'):
                scale = min(3.0, scale + 0.1)  # Increase scale
        elif key == ord('-') or key == ord('_'):
                scale = max(1.0, scale - 0.1)  # Decrease scale

    if translation_on:
        tx, ty = 500, 300
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        processed = cv2.warpAffine(processed, translation_matrix, (w, h))

    if rotation_on:
        center = (w // 2, h // 2)
        rotation_angle = 45
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        processed = cv2.warpAffine(processed, rotation_matrix, (w, h))

    if perspective_on:
        p1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        p2 = np.float32([[0, 0], [w-1, 0], [w*0.2, h-1], [w*0.8, h-1]])
        perspective_matrix = cv2.getPerspectiveTransform(p1, p2)
        processed = cv2.warpPerspective(processed, perspective_matrix, (w, h))
    
    # Combine original and processed images side by side
    combined = cv2.hconcat([frame, cv2.resize(processed, (w, h))])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Processed', (w+10, 30), font, 1, (255, 255, 255), 2)
    
    # Display FPS for both frames
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(combined, fps_text, (10, 70), font, 1, (0, 255, 0), 3)  # FPS for the original frame (left-top)
    cv2.putText(combined, fps_text, (w + 10, 70), font, 1, (0, 255, 0), 3)  # FPS for the processed frame (right-top)

    # Add status indicators in the middle lower part
    status_y = h - 60
    status_x = w // 2  # Center horizontally

    # Display all indicators with their status
    draw_status_indicator(combined, 'T: ', "On" if translation_on else "Off", translation_on, (status_x - 250, status_y))
    draw_status_indicator(combined, 'R: ', "On" if rotation_on else "Off", rotation_on, (status_x - 100, status_y))
    draw_status_indicator(combined, 'S: ', f"{scale:.1f}x", scaling_on, (status_x + 50, status_y))
    draw_status_indicator(combined, 'P: ', "On" if perspective_on else "Off", perspective_on, (status_x + 200, status_y))
    draw_status_indicator(combined, 'Q: ', "Exit", False, (status_x + 350, status_y))  # Adding Q as Exit indicator

    # Display the resulting frame
    cv2.imshow('Original and Processed', combined)
    
    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    
    # Toggle transformations based on key press
    if key == ord('t'):
        translation_on = not translation_on
        rotation_on = False
        scaling_on = False
        perspective_on = False
    elif key == ord('r'):
        rotation_on = not rotation_on
        translation_on = False
        scaling_on = False
        perspective_on = False
    elif key == ord('s'):
        scaling_on = not scaling_on
        translation_on = False
        rotation_on = False
        perspective_on = False
    elif key == ord('p'):
        perspective_on = not perspective_on
        translation_on = False
        rotation_on = False
        scaling_on = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
