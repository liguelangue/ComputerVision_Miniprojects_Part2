import cv2
import os
import argparse
import numpy as np

def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_files]
    return images

def determine_horizontal_position(kp1, kp2, matches):
    avg_x1 = np.mean([kp1[m.queryIdx].pt[0] for m in matches])
    avg_x2 = np.mean([kp2[m.trainIdx].pt[0] for m in matches])
    width1 = max(kp.pt[0] for kp in kp1)
    width2 = max(kp.pt[0] for kp in kp2)
    ratio1 = avg_x1 / width1
    ratio2 = avg_x2 / width2
    return ratio1 > ratio2

def determine_vertical_position(img1, img2, kp1, kp2, matches):
    avg_y1 = np.mean([kp1[m.queryIdx].pt[1] for m in matches])
    avg_y2 = np.mean([kp2[m.trainIdx].pt[1] for m in matches])
    height1 = img1.shape[0]
    height2 = img2.shape[0]
    ratio1 = avg_y1 / height1
    ratio2 = avg_y2 / height2
    return ratio1 > ratio2

def order_images(images):
    sift = cv2.SIFT_create()
    keypoints_descriptors = [sift.detectAndCompute(img, None) for img in images]
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    match_results = {}
    match_details = {}

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            kp1, des1 = keypoints_descriptors[i]
            kp2, des2 = keypoints_descriptors[j]
            matches = bf.match(des1, des2)
            match_results[(i, j)] = len(matches)
            match_details[(i, j)] = (kp1, kp2, matches)

    sorted_matches = sorted(match_results.items(), key=lambda x: x[1], reverse=True)
    (first_pair, _) = sorted_matches[0]
    first_image_index, second_image_index = first_pair
    kp1, kp2, matches = match_details[first_pair]
    should_be_left = determine_horizontal_position(kp1, kp2, matches)

    pair1_indices = []
    if should_be_left:
        pair1_indices = [first_image_index, second_image_index]
    else:
        pair1_indices = [second_image_index, first_image_index]

    remaining_indices = [i for i in range(len(images)) if i != first_image_index and i != second_image_index]
    second_pair_match_results = {}

    for i in range(len(remaining_indices)):
        for j in range(i + 1, len(remaining_indices)):
            idx1, idx2 = remaining_indices[i], remaining_indices[j]
            kp1, des1 = keypoints_descriptors[idx1]
            kp2, des2 = keypoints_descriptors[idx2]
            matches = bf.match(des1, des2)
            second_pair_match_results[(idx1, idx2)] = len(matches)
            match_details[(idx1, idx2)] = (kp1, kp2, matches)

    (sorted_second_pair, _) = sorted(second_pair_match_results.items(), key=lambda x: x[1], reverse=True)[0]
    third_image_index, fourth_image_index = sorted_second_pair
    kp1, kp2, matches = match_details[sorted_second_pair]
    should_be_left = determine_horizontal_position(kp1, kp2, matches)

    pair2_indices = []
    if should_be_left:
        pair2_indices = [third_image_index, fourth_image_index]
    else:
        pair2_indices = [fourth_image_index, third_image_index]

    top_img = cv2.hconcat([images[pair1_indices[0]], images[pair1_indices[1]]])
    bottom_img = cv2.hconcat([images[pair2_indices[0]], images[pair2_indices[1]]])

    kp_top, des_top = sift.detectAndCompute(top_img, None)
    kp_bottom, des_bottom = sift.detectAndCompute(bottom_img, None)
    matches_vertical = bf.match(des_top, des_bottom)
    should_be_top = determine_vertical_position(top_img, bottom_img, kp_top, kp_bottom, matches_vertical)

    ordered_indices = []
    if should_be_top:
        ordered_indices = pair1_indices + pair2_indices
    else:
        ordered_indices = pair2_indices + pair1_indices

    ordered_images = [images[i] for i in ordered_indices]
    return ordered_images

def stitch_pair(img1, img2, direction='horizontal'):
    MIN_MATCH_COUNT = 10

    # Detect keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get the shape of input images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Get the canvas dimensions
        # Get the corner points of both images
        corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        # Transform the corners of img1 using the homography
        transformed_corners_img1 = cv2.perspectiveTransform(corners_img1, M)

        # Combine the corners to get the bounding box
        all_corners = np.concatenate((transformed_corners_img1, corners_img2), axis=0)

        # Find the bounding box of all corners
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

        # Compute the translation to shift the images
        translation = np.array([[1, 0, -xmin],
                                [0, 1, -ymin],
                                [0, 0, 1]])

        # Warp img1
        result_width = xmax - xmin
        result_height = ymax - ymin
        result = cv2.warpPerspective(img1, translation.dot(M), (result_width, result_height))

        # Paste img2 into the result image
        result[-ymin:h2 - ymin, -xmin:w2 - xmin] = img2

        return result
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
        return None

def stitch_images(ordered_images):
    # Stitch first pair horizontally
    top_image = stitch_pair(ordered_images[0], ordered_images[1], direction='horizontal')
    if top_image is None:
        print("Failed to stitch top images")
        return None

    # Stitch second pair horizontally
    bottom_image = stitch_pair(ordered_images[2], ordered_images[3], direction='horizontal')
    if bottom_image is None:
        print("Failed to stitch bottom images")
        return None

    # Stitch top and bottom images vertically
    final_image = stitch_pair(top_image, bottom_image, direction='vertical')
    if final_image is None:
        print("Failed to stitch top and bottom images")
        return None

    return final_image

def combine_images_into_grid(images, rows=2, cols=2):
    if not images:
        print("No images found in the directory.")
        return None

    # Resize images to a standard size for uniform display
    max_height, max_width = 150, 150
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]

    # Create blank images to complete the grid if necessary
    blank_image = np.zeros_like(resized_images[0])
    while len(resized_images) < rows * cols:
        resized_images.append(blank_image)

    # Arrange images into rows and columns
    rows_of_images = [np.hstack(resized_images[i * cols:(i + 1) * cols]) for i in range(rows)]
    combined_image = np.vstack(rows_of_images)

    return combined_image

def display_stitched_image(image):
    if image is None:
        return

    cv2.imshow('Stitched Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_image(image):
    """Crop the unnecessary areas (e.g., black borders) from the stitched image."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to create a mask of non-black areas
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the largest contour
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image
    else:
        return image  # Return the original image if no contours are found

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stitch 9 images and display.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('-o', '--output', type=str, help='Path to save the stitched image', default='stitched_output.jpg')
    args = parser.parse_args()

    # Read images from the given folder
    images = read_images_from_folder(args.directory)

    # Once you start, ignore combine_images_into_grid()
    # Instead, complete stitch_image(), if needed you can add more arguments
    # at stitch_image()
    # stitched_image = stitch_image(images) 

    # Sort and arrange the images based on feature matching
    ordered_images = order_images(images)

    # Display the ordered images as a grid
    # stitched_image = combine_images_into_grid(ordered_images, rows=2, cols=2)
    stitched_image = stitch_images(ordered_images)

    stitched_image = crop_image(stitched_image)
    
    # Display the stitched image
    display_stitched_image(stitched_image)

    cv2.imwrite(os.path.join(args.directory, f"{os.path.basename(os.path.normpath(args.directory))}.jpg"), stitched_image)


if __name__ == '__main__':
    main()