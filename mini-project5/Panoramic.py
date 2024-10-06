import cv2
import numpy as np
import time

def capture_frames():
    """
    Capture frames from the camera for panorama creation.
    """
    cap = cv2.VideoCapture(0) # Initialize the camera
    frames = [] # Variables for panorama capture
    is_capturing = False # Variables for panorama capture

    while True:
        ret, frame = cap.read() # Capture frame-by-frame
        if not ret:
            break
        
        cv2.imshow('Frame', frame)  # Display the frame
        key = cv2.waitKey(1) & 0xFF # Wait for key press
        
        # Start capturing frames when 's' is pressed
        if key == ord('s'):
            is_capturing = True
            print("Started capturing frames for panorama")
        
        # Capture frames for panorama
        if is_capturing:
            frames.append(frame)
            # print(f"Captured frame {len(frames)}")
        
        # Stop capturing when 'a' is pressed
        if key == ord('a'):
            is_capturing = False
            print(f"Stopped capturing. Total frames: {len(frames)}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames

def SIFT(image):
    """
    Detect and describe features in the image using SIFT.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Match features between two images.
    """
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def create_panorama(frames):
    """
    Create a panorama from a list of frames.
    """
    # Select only 2 frames: the first and the last
    selected_frames = [frames[0], frames[-1]]
    
    if len(selected_frames) < 2:
        return selected_frames[0] if selected_frames else None
    
    # Detect and describe features
    kp1, desc1 = SIFT(selected_frames[0])
    kp2, desc2 = SIFT(selected_frames[1])
    
    # Match features
    matches = match_features(desc1, desc2)
    
    # Display detected features and matches
    img1_with_kp = cv2.drawKeypoints(selected_frames[0], kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_kp = cv2.drawKeypoints(selected_frames[1], kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('Keypoints in First Frame', img1_with_kp)
    cv2.imshow('Keypoints in Last Frame', img2_with_kp)
    cv2.waitKey(1000)  # Display for 1 second
    
    img_matches = cv2.drawMatches(selected_frames[0], kp1, selected_frames[1], kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(1000)  # Display for 1 second
    
    print(f"Number of good matches between the two frames: {len(matches)}")
    
    # Stitch images
    if len(matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = selected_frames[0].shape[:2]
        warped_panorama = cv2.warpPerspective(selected_frames[0], M, (w * 2, h))
        
        warped_panorama[0:selected_frames[1].shape[0], 0:selected_frames[1].shape[1]] = selected_frames[1]
        
        return warped_panorama
    else:
        print("Not enough matches found between the two frames")
        return None


if __name__ == "__main__":
    print("Press 's' to start capturing, 'a' to stop capturing")
    captured_frames = capture_frames()
    
    if captured_frames:
        print("Creating panorama...")
        final_panorama = create_panorama(captured_frames)
        
        if final_panorama is not None:
            cv2.imshow('Panorama', final_panorama)
            print("successfully create panorama")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite('panorama.jpg', final_panorama)
            # print("Panorama saved as 'panorama.jpg'")
        else:
            print("Failed to create panorama")
    else:
        print("No frames captured")
