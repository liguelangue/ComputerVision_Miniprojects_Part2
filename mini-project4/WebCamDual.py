import cv2

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

while True:
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

    # Stack the frames from both cameras horizontally
    combined = cv2.hconcat([frame1, frame2])
    
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
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            matches = bf.match(dp1, dp2)
            matches = sorted(matches, key=lambda x: x.distance)

            max_matches = 5
            combined = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:max_matches],
                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.imshow('Matching Results', combined)
    else:
        # Stack the frames from both cameras horizontally
        combined = cv2.hconcat([frame1, frame2])

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

# Release the cameras and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
