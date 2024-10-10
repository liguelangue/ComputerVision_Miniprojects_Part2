import cv2
import numpy as np
import time

class PanoramaGenerator:
    def __init__(self, camera_index=0, frame_interval=1):
        # Initialize the video capture, frame list, and other necessary variables
        self.cap = cv2.VideoCapture(camera_index)
        self.frames = []
        self.is_capturing = False
        self.last_capture_time = 0
        self.frame_interval = frame_interval

    def capture_frames(self):
        # Capture frames from the camera based on user input
        print("Press 's' to start capturing, 'a' to stop capturing")
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = time.time()
            frame_count += 1
            fps = frame_count / (current_time - start_time)

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF

            # Start capturing frames when 's' key is pressed
            if key == ord('s'):
                self.is_capturing = True
                print("Started capturing frames for panorama")

            # Capture frames at the specified interval
            if self.is_capturing and current_time - self.last_capture_time >= self.frame_interval:
                self.frames.append(frame)
                self.last_capture_time = current_time
                print(f"Captured frame {len(self.frames)}")

            # Stop capturing when 'a' key is pressed
            if key == ord('a'):
                self.is_capturing = False
                print(f"Stopped capturing. Total frames: {len(self.frames)}")
                break

        # Release the camera and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def SIFT(self, image):
        # Convert the image to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create a SIFT detector to find keypoints and descriptors
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        # Use BFMatcher to find matches between descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        # Sort matches by distance (lower distance is better)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def blend_images(self, img1, img2, homography):
        # Get the dimensions of both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Determine the corners of img2 after applying the homography
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners_img2 = cv2.perspectiveTransform(corners_img2, homography)

        # Determine the corners of img1
        corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        # Combine corners from both images to determine the final bounding box
        all_corners = np.vstack((corners_img1, warped_corners_img2))

        # Calculate the minimum and maximum x, y coordinates for the bounding box
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Translation distances to make sure all parts are in positive coordinates
        translation_dist = [-xmin, -ymin]

        # Determine the size of the resulting panorama
        panorama_width = xmax - xmin
        panorama_height = ymax - ymin

        # Create a translation matrix to shift the panorama
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        # Warp img2 into the panorama space
        warped_img2 = cv2.warpPerspective(img2, H_translation @ homography, (panorama_width, panorama_height))

        # Create the result image and place img1 in it
        result = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        result[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1] = img1

        # Create masks to identify overlapping regions
        mask1 = np.ones((h1, w1), dtype=np.uint8) * 255
        mask2 = np.ones((h2, w2), dtype=np.uint8) * 255
        warped_mask2 = cv2.warpPerspective(mask2, H_translation @ homography, (panorama_width, panorama_height))

        # Find overlapping areas and blend them
        overlap_mask = (warped_mask2 > 0) & (np.sum(result, axis=2) > 0)
        result[overlap_mask] = cv2.addWeighted(result[overlap_mask], 0.5, warped_img2[overlap_mask], 0.5, 0)
        # Add non-overlapping regions from warped_img2
        non_overlap_mask = (warped_mask2 > 0) & (np.sum(result, axis=2) == 0)
        result[non_overlap_mask] = warped_img2[non_overlap_mask]

        return result

    def stitch_two_images(self, img1, img2):
        # Detect features and compute descriptors for both images using SIFT
        kp1, desc1 = self.SIFT(img1)
        kp2, desc2 = self.SIFT(img2)

        # Match features between the two images
        matches = self.match_features(desc1, desc2)

        # If there are enough matches, proceed with stitching
        if len(matches) > 50:
            # Extract the matched keypoints
            src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Compute the homography matrix using RANSAC to filter outliers
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            inliers_count = sum(matches_mask)

            # If there are enough inliers, blend the images
            if inliers_count > 2:
                print(f"Number of inliers: {inliers_count}")
                panorama = self.blend_images(img1, img2, M)
                return panorama
            else:
                print("Not enough inliers to reliably stitch the images")
                return None
        else:
            print("Not enough matches found to stitch the images")
            return None

    def create_panorama(self):
        # If there are fewer than two frames, return the first frame or None
        if len(self.frames) < 2:
            return self.frames[0] if self.frames else None

        # Start with the first frame and progressively stitch each additional frame
        panorama = self.frames[0]
        for i in range(1, len(self.frames)):
            panorama = self.stitch_two_images(panorama, self.frames[i])
            if panorama is None:
                print(f"Stitching failed between frame {i} and {i + 1}")
                return None

        return panorama

    def run(self):
        # Capture frames from the camera
        self.capture_frames()
        if self.frames:
            print("Creating panorama...")
            final_panorama = self.create_panorama()

            # If panorama creation is successful, crop and display it
            if final_panorama is not None:
                # Crop central 50% height and central 80% width
                h, w = final_panorama.shape[:2]
                cropped_panorama = final_panorama[int(0.25 * h):int(0.75 * h), int(0.1 * w):int(0.9 * w)]

                cv2.imshow('Panorama', cropped_panorama)
                print("Successfully created panorama")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # Optionally, save the panorama
                cv2.imwrite('panorama.jpg', cropped_panorama)
                print("Panorama saved as 'panorama.jpg'")
            else:
                print("Failed to create panorama")
        else:
            print("No frames captured")

if __name__ == "__main__":
    panorama_generator = PanoramaGenerator()
    panorama_generator.run()
