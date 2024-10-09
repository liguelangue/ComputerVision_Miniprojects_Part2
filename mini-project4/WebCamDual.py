import cv2
import numpy as np
import time

def main(use_video=False, video1_path=None, video2_path=None):
    # Initialize the cameras or video files
    if use_video:
        if video1_path is None or video2_path is None:
            print("Please provide valid paths for both video files.")
            return
        
        cap1 = cv2.VideoCapture(video1_path)  # First video file
        cap2 = cv2.VideoCapture(video2_path)  # Second video file
    else:
        cap1 = cv2.VideoCapture(0)  # First camera
        cap2 = cv2.VideoCapture(1)  # Second camera, change the index if necessary

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Initialize SIFT
    sift_on = False
    sift = cv2.SIFT_create(500)

    # Initialize ORB
    orb_on = False
    orb = cv2.ORB_create(nfeatures=1000)

    matching_on = False
    orb_matching_on = False

    # Initialize Brute-Force Matchers
    bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Fixed thresholds
    threshold_sift = 500
    threshold_orb = 500

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while True:
        frame_start_time = time.time()

        cap1.grab()
        cap2.grab()
        # Capture frame-by-frame from the first source (camera or video)
        ret1, frame1 = cap1.read()
        # Capture frame-by-frame from the second source (camera or video)
        ret2, frame2 = cap2.read()

        # Check if frames are captured (or video has ended)
        if not ret1 or not ret2:
            print("One of the streams ended or failed to capture.")
            break

        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # GRAY
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        # Keypoints & Descriptors
        kp1, dp1, kp2, dp2 = None, None, None, None

        if sift_on or matching_on:
            # SIFT keypoints and descriptors
            kp1, dp1 = sift.detectAndCompute(gray1, None)
            kp2, dp2 = sift.detectAndCompute(gray2, None)
        elif orb_on or orb_matching_on:
            # ORB keypoints and descriptors
            kp1, dp1 = orb.detectAndCompute(gray1, None)
            kp2, dp2 = orb.detectAndCompute(gray2, None)

        # Visualization
        method_name = ''
        if sift_on:
            method_name = 'SIFT Keypoints'
            frame1_kp1 = cv2.drawKeypoints(frame1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            frame2_kp2 = cv2.drawKeypoints(frame2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            combined = cv2.hconcat([frame1_kp1, frame2_kp2])
        elif orb_on:
            method_name = 'ORB Keypoints'
            frame1_kp1 = cv2.drawKeypoints(frame1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            frame2_kp2 = cv2.drawKeypoints(frame2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            combined = cv2.hconcat([frame1_kp1, frame2_kp2])
        elif matching_on or orb_matching_on:
            if dp1 is not None and dp2 is not None and kp1 is not None and kp2 is not None:
                if matching_on:
                    method_name = 'SIFT Matching'
                    matches = bf_sift.match(dp1, dp2)
                    threshold = threshold_sift
                elif orb_matching_on:
                    method_name = 'ORB Matching'
                    matches = bf_orb.match(dp1, dp2)
                    threshold = threshold_orb
                matches = sorted(matches, key=lambda x: x.distance)
                # Get good matches
                good_matches = [m for m in matches if m.distance < threshold]
                # Calculate matching score
                total_keypoints = min(len(kp1), len(kp2))
                matching_score = len(good_matches) / total_keypoints if total_keypoints > 0 else 0
                # Calculate average distance of good matches
                avg_distance = np.mean([m.distance for m in good_matches]) if good_matches else 0
                # Draw top matches
                combined = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches[:10], None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # Add text annotations
                cv2.putText(combined, f"Matching Score: {matching_score:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, f"Avg Distance: {avg_distance:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, f"Keypoints detected: {len(kp1)}/{len(kp2)}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, f"Good Matches: {len(good_matches)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Stack the frames from both sources horizontally
            combined = cv2.hconcat([frame1, frame2])

        # Display method name
        if method_name:
            cv2.putText(combined, method_name, (combined.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(combined, f"FPS: {fps:.2f}", (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frames
        cv2.imshow('Video/Camera 1 and Video/Camera 2', combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            sift_on = not sift_on
            matching_on = False
            orb_on = False
            orb_matching_on = False

        elif key == ord('m'):
            matching_on = not matching_on
            sift_on = False
            orb_on = False
            orb_matching_on = False

        elif key == ord('o'):
            orb_on = not orb_on
            sift_on = False
            matching_on = False
            orb_matching_on = False

        elif key == ord('n'):
            orb_matching_on = not orb_matching_on
            orb_on = False
            sift_on = False
            matching_on = False

        elif key == ord('q'):
            break

        # Calculate processing time for this frame
        frame_time = time.time() - frame_start_time
        # If processing is faster than 30 FPS, wait the remaining time
        if frame_time < 1/30:
            time.sleep(1/30 - frame_time)

    # Release the sources and close all OpenCV windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Request user input for whether to use video files
    use_video_input = input("Would you like to use video files? (y/n): ").strip().lower()
    if use_video_input == 'y':
        video1_path = input("Enter the path for the first video: ").strip()
        video2_path = input("Enter the path for the second video: ").strip()
        main(use_video=True, video1_path=video1_path, video2_path=video2_path)
    else:
        main(use_video=False)