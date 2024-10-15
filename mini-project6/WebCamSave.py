# USAGE: python WebCamSave.py -f video_file_name -o out_video.avi

# import the necessary packages
import cv2
import numpy as np
import time
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")

args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width  = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

def region_of_interest(image):
    height, width = image.shape
    polygon = np.array([
    [(0, height), (width, height), (int(0.55*width), int(0.6*height)), (int(0.45*width), int(0.6*height))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Add your code HERE: For example,
    #cv2.putText(frame, "Lane Detection", (int(width/4), int(height/2)),
    #            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

    # Convert each frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply ROI
    roi_image = region_of_interest(edges)

    # Write the frame to the output video file
    if args.out:
        out.write(roi_image)

    # show the output frame
    cv2.imshow("Frame", roi_image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the video capture object
vs.release()
out.release()
cv2.destroyAllWindows()
