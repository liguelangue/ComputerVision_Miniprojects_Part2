import cv2
import numpy as np
import time

class VideoProcessor:
    def __init__(self, input_file=None, output_file="output.avi", rotate_angle=0):
        # Initialize video capture from file or webcam
        if input_file:
            self.vs = cv2.VideoCapture(input_file)
        else:
            self.vs = cv2.VideoCapture(0)

        # Allow the camera to warm up
        time.sleep(2.0)

        # Get video resolution
        self.width = int(self.vs.get(3))
        self.height = int(self.vs.get(4))

        # Video writer to output the processed frames
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(output_file, fourcc, 20.0, (self.width, self.height), True)

        # Store rotation angle
        self.rotate_angle = rotate_angle

        # Controls
        self.detection_running = False
        self.paused = False
        self.show_fps = input_file is None

    def rotate_frame(self, frame, angle):
        """
        Rotate the frame by the specified angle.
        """
        if angle != 0:
            # Get the rotation matrix
            center = (self.width // 2, self.height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Perform the rotation
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (self.width, self.height))
            return rotated_frame
        return frame

    def rotate_points(self, points, angle):
        """
        Rotate points by the specified angle around the center of the frame.
        """
        if angle == 0 or points is None:
            return points

        # Get the rotation matrix
        center = (self.width // 2, self.height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply the rotation matrix to each point
        rotated_points = []
        for point in points:
            x1, y1, x2, y2 = point
            p1 = np.dot(rotation_matrix, np.array([x1, y1, 1]))
            p2 = np.dot(rotation_matrix, np.array([x2, y2, 1]))
            rotated_points.append([int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])])

        return rotated_points

    def filter_white(self, frame):
        """
        Apply a color filter to keep only white parts of the frame.
        """
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the average brightness to adjust the white threshold dynamically
        brightness = np.mean(hsv[:, :, 2])  # The V channel represents brightness

        # Define the lower and upper boundaries for white color in HSV space
        lower_white_v = max(190, brightness - 70)
        upper_white = np.array([180, 30, 255])
        lower_white_full = np.array([0, 0, lower_white_v])

        # Create a mask to filter out all colors except white
        mask = cv2.inRange(hsv, lower_white_full, upper_white)

        # Apply the mask to the original frame
        filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return filtered_frame

    def preprocess(self, frame):
        """
        Preprocess the frame by converting to grayscale, blurring, and edge detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def region_of_interest(self, img):
        """
        Apply a mask to keep only the region of interest (trapezoid).
        """
        mask = np.zeros_like(img)   
        #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        height, width = img.shape
        polygon = np.array([[
            (0, height),
            (width, height),
            (int(width * 0.90), int(height * 0.6)),
            (int(width * 0.01), int(height * 0.6))
        ]], np.int32)

        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygon, ignore_mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def filter_lines(self, lines):
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x2 - x1) == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if slope < -0.5:
                left_lines.append(line)
            elif slope > 0.5:
                right_lines.append(line)

        return left_lines, right_lines

    def average_slope_intercept(self, lines):
        """
        Average the slope and intercept of the left and right lines and return a single line for each side.
        """
        if len(lines) == 0:
            return None
        
        slopes = []
        intercepts = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            slopes.append(slope)
            intercepts.append(intercept)

        slope_avg = np.mean(slopes)
        intercept_avg = np.mean(intercepts)
        return slope_avg, intercept_avg

    def make_line(self, slope, intercept, height):
        """
        Convert slope and intercept into pixel coordinates for drawing the line.
        """
        if slope == 0 or np.isnan(slope) or np.isinf(slope):
            return None
        y1 = height
        y2 = int(y1 * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    
    def draw_lines(self, frame, lines, color=(0, 255, 0), thickness=5):
        """
        Draw multiple lines on the frame.
        """
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

    def process_frame(self, frame):
        """
        Process a single frame by applying color filtering, edge detection, Hough line transform,
        filtering lines, and extending them from the top to the bottom of the video.
        """
        # Create a copy of the original frame for processing
        frame_copy = frame.copy()

        # Apply rotation, filtering, and edge detection on the copy
        rotated_frame = self.rotate_frame(frame_copy, self.rotate_angle)
        filtered_frame = self.filter_white(rotated_frame)
        edges = self.preprocess(filtered_frame)
        roi = self.region_of_interest(edges)
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)

        # Draw the final lines on the original frame
        if lines is not None:
            left_lines, right_lines = self.filter_lines(lines)

            left_slope_intercept = self.average_slope_intercept(left_lines)
            right_slope_intercept = self.average_slope_intercept(right_lines)

            lines_to_draw = []

            if left_slope_intercept is not None:
                left_line = self.make_line(left_slope_intercept[0], left_slope_intercept[1], self.height)
                if left_line is not None:
                    lines_to_draw.append(left_line)

            if right_slope_intercept is not None:
                right_line = self.make_line(right_slope_intercept[0], right_slope_intercept[1], self.height)
                if right_line is not None:
                    lines_to_draw.append(right_line)

            # Rotate the lines back to the original orientation
            rotated_lines = self.rotate_points(lines_to_draw, -self.rotate_angle)

            # Draw the rotated lines on the original frame
            if rotated_lines is not None:
                self.draw_lines(frame, rotated_lines, color=(0, 0, 255))

        return frame
    
    def display_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def start(self):
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = self.vs.read()
            if not ret:
                break
            
            if self.detection_running and not self.paused:
                processed_frame = self.process_frame(frame)
                frame_count += 1

                font_scale = 1.5
                font_thickness = 3
                detection_text = 'Detection Started'
                text_size = cv2.getTextSize(detection_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = frame.shape[1] - text_size[0] - 10
                text_y = 40
                cv2.putText(frame, detection_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                if self.show_fps:

                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    self.display_fps(processed_frame, fps)

                self.out.write(processed_frame)
                cv2.imshow("Frame", processed_frame)
            
            else:
                if self.paused:
                    detection_text = 'Detection Paused'
                    cv2.putText(frame, detection_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.detection_running = True
                print('Detection started')
            elif key == ord('p'):
                self.paused = not self.paused
                print('Detection paused' if self.paused else 'Detection resumed')
            elif key == ord('q'):
                break

        self.vs.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video file path or camera input")
    parser.add_argument("-f", "--file", type=str, help="Path to the video file")
    parser.add_argument("-o", "--out", type=str, help="Output video file name", default="output.avi")
    parser.add_argument("-r", "--rotate", type=float, help="Rotation angle in degrees", default=10)
    args = parser.parse_args()

    video_processor = VideoProcessor(input_file=args.file, output_file=args.out, rotate_angle=args.rotate)
    video_processor.start()