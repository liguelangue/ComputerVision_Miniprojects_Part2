# CS5330-Mid-Term-Fall2024
## Group 8: Team Members
Xinmeng Wu, Pingyi Xu, Anning Tian, Qinhao Zhang
## Instructions: how to run the code
```
python tile-quiz.py -d ./data-3
```
'-d': Path to the folder containing images.
## Description of Approach and Solution
### 1. Image Positioning
Extract Keypoints: Used SIFT to detect keypoints and descriptors in each image.

Match Keypoints: Matched keypoints between image pairs using BFMatcher with L2 norm.

Determine Horizontal Order: Calculated average x-coordinates of matched keypoints for each pair. Compared these averages relative to image widths to decide which image should be on the left.

Determine Vertical Order: After combining horizontally ordered pairs, calculated average y-coordinates relative to image heights. Decided which pair should be on top.

Final Image Ordering: Combined the two horizontally ordered pairs based on the determined vertical order.
### 2. Stitching
Detect Keypoints: Used SIFT to extract keypoints and descriptors from a pair of images.

Match Keypoints: Matched descriptors using BFMatcher and applied a ratio test to select good matches.

Calculate Homography: If enough good matches are found, computed the homography matrix using matched keypoints.

Transform and Warp: Used the homography to warp the first image and find the bounding box of the combined images.

Combine Images: Pasted the second image into the warped result to create a seamless stitched image.

Stitch All Pairs: Repeated this process to stitch horizontally aligned image pairs, then vertically combined them to form the final stitched image.
### 3. Post-Processing
The crop_image function removes unnecessary areas like black borders from the image by converting it to grayscale, applying a binary threshold to detect non-black regions, and cropping the image based on the largest contour found. If no contours are detected, it returns the original image.

## Keypoint Matching Algorithm and Stitching Method
Keypoint Matching Algorithm: The approach leverages SIFT (Scale-Invariant Feature Transform) to detect and describe keypoints in the images. Keypoints are then matched using BFMatcher with L2 norm. To filter out weaker matches, a ratio test is applied, retaining only those matches where the distance of the closest descriptor is significantly smaller than the second-closest.

Stitching Method: The stitching is performed by calculating a homography matrix between matched keypoints. The homography describes the perspective transformation needed to align the images. This matrix is used to warp one image to the perspective of the other, and the two images are combined to form a seamless final image.