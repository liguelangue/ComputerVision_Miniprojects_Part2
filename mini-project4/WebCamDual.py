import cv2
import numpy as np
import time

# Initialize the cameras
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(1)  # Second camera, change the index if necessary

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

sift_on = False
sift = cv2.SIFT_create(500)

matching_on = False

# Initialize Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Fixed threshold
threshold = 500

# Variables for FPS calculation
start_time = time.time()
frame_count = 0

while True:
    frame_start_time = time.time()
    
    cap1.grab()
    cap2.grab()
    # Capture frame-by-frame from the first camera
    ret1, frame1 = cap1.read()
    # Capture frame-by-frame from the second camera
    ret2, frame2 = cap2.read()

    # Check if frames are captured
    if not ret1 or not ret2:
        break

    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # GRAY
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Keypoints & Descriptors
    kp1, dp1 = sift.detectAndCompute(gray1, None)
    kp2, dp2 = sift.detectAndCompute(gray2, None)

    # SIFT
    if sift_on:
        frame1_kp1 = cv2.drawKeypoints(frame1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame2_kp2 = cv2.drawKeypoints(frame2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        combined = cv2.hconcat([frame1_kp1, frame2_kp2])

    # Features Matching
    elif matching_on:
        if dp1 is not None and dp2 is not None:

            matches = bf.match(dp1, dp2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Get good matches
            good_matches = [m for m in matches if m.distance < threshold]

            # Calculate matching score
            total_keypoints = min(len(kp1), len(kp2))
            matching_score = len(good_matches) / total_keypoints if total_keypoints > 0 else 0
            
            # Calculate average distance of good matches, help to set threshold
            avg_distance = np.mean([m.distance for m in good_matches]) if good_matches else 0
            
            # Draw top 10 matches
            combined = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches[:10], None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.putText(combined, f"Matching Score: {matching_score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"Avg Distance: {avg_distance:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"Keypoints detected: {len(kp1)}/{len(kp2)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"Good Matches: {len(good_matches)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # Stack the frames from both cameras horizontally
        combined = cv2.hconcat([frame1, frame2])

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(combined, f"FPS: {fps:.2f}", (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frames
    cv2.imshow('Camera 1 and Camera 2', combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        sift_on = not sift_on
        matching_on = False
    
    if key == ord('m'):
        matching_on = not matching_on
        sift_on = False

    # Break the loop on 'q' key press
    elif key == ord('q'):
        break

    # Calculate processing time for this frame
    frame_time = time.time() - frame_start_time
    # If processing is faster than 30 FPS, wait the remaining time
    if frame_time < 1/30:
        time.sleep(1/30 - frame_time)

# Release the cameras and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
