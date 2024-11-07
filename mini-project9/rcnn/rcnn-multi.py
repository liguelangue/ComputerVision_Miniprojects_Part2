import os
import cv2
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.utils import shuffle



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA support.")
else:
    print("TensorFlow is not built with CUDA support.")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Path to dataset (base directory)
base_dir = r"D:\NEU\CS5330\mini_proj_9\9-1\data_V3"
print("base_dir found")

# Define categories and assign each a label
categories = [ "Glasses", "Phones", "TV"]
category_labels = {category: idx for idx, category in enumerate(categories)}
print("base_cate defined")

# Lists to store data
train_images = []
train_labels = []
train_filenames = []  
# svm_images = []
# svm_labels = []

# IoU calculation
def get_iou(bb1, bb2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return max(0.0, min(1.0, iou))

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png")
print("extension set")


# Step 1: Running Selective Search on individual images to obtain region proposals (2000 here).
# Enable optimized computation in OpenCV
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
print("Data directory content:", os.listdir(base_dir))
# Iterate over categories to load images and labels
# Iterate over categories to load images and labels
# Iterate over categories to load images and labels
for category in categories:
    category_dir = os.path.join(base_dir, category)
    label_dir = os.path.join(base_dir, f"{category}_annotations")
    label = category_labels[category]

    for label_file in os.listdir(label_dir):
        try:
            counter = 0
            falsecounter = 0
            filename = label_file.split(".")[0]
            image_file = next((f for f in os.listdir(category_dir) if f.startswith(filename) and f.endswith(image_extensions)), None)
            if image_file is None:
                print(f"No matching image file found for {label_file}")
                continue
            
            image_path = os.path.join(category_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image {image_file}")
                continue
            else:
                print(f"Loaded image {image_file} successfully")
                train_filenames.append(filename)  # Only add if image is successfully loaded

            # Store the original dimensions
            original_height, original_width = image.shape[:2]

            # Resize image to 256x256
            resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

            # Calculate scaling factors
            x_scale = 256 / original_width
            y_scale = 256 / original_height

            # Read bounding box annotations and adjust to new size
            df = pd.read_csv(os.path.join(label_dir, label_file))
            gtvalues = []
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, row.iloc[0].split(" "))
                
                # Scale bounding box coordinates
                x1 = int(x1 * x_scale)
                y1 = int(y1 * y_scale)
                x2 = int(x2 * x_scale)
                y2 = int(y2 * y_scale)

                gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

            print(f"Processing category: {category} with {len(os.listdir(category_dir))} images.")

            # Run selective search on resized image
            ss.setBaseImage(resized_image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            max_proposals = 2000  # 可根据需求调整数量
            print(f"Number of search results before filtering: {len(ssresults)}")
            ssresults = ssresults[:max_proposals]
            print("Number of search results:", len(ssresults))

            print(f"Current positive examples: {counter}, Current negative examples: {falsecounter}")

            # Initialize counters for positive and negative examples
            counter = 0
            falsecounter = 0

            # Step 2: Classify region proposals as positive and negative examples based on IoU.
            for e, result in enumerate(ssresults):
                x, y, w, h = result

                # Iterate through ground truth boxes to calculate IoU
                for gtval in gtvalues:
                    iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    timage = resized_image[y:y + h, x:x + w]

                    # Positive example
                    if counter < 30 and iou > 0.7:
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(label)
                        # train_filenames.extend([filename] * 1) 
                        counter += 1

                    # Negative example
                    elif falsecounter < 30 and iou < 0.3:
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(len(categories))
                        # train_filenames.extend([filename] * 1)  
                        falsecounter += 1

                if counter >= 30 and falsecounter >= 30:
                    break
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        continue



# Conversion of train data into arrays for further training
X_new = np.array(train_images)
Y_new = np.array(train_labels)




# Path to test dataset
test_dir = r"D:\NEU\CS5330\mini_proj_9\9-1\data_V3_test"
test_images = []
test_labels = []
test_filenames = []

# Load test dataset
# Load test dataset
for category in categories:
    category_dir = os.path.join(test_dir, category)
    label_dir = os.path.join(test_dir, f"{category}_annotations")
    label = category_labels[category]

    for image_file in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_file}")
            continue
        
        # Resize image to ensure consistent shape
        resized_image = cv2.resize(image, (224, 224))  # Adjust size to match model input
        test_images.append(resized_image)
        test_labels.append(label)
        test_filenames.append(image_file)  # Store the filename for future use

# Convert test data into numpy arrays
X_test = np.array(test_images)
Y_test = np.array(test_labels)




print("step1 done")

# Step 3: Passing every proposal through a pretrained network (VGG16 trained on ImageNet) to output a fixed-size feature vector (4096 here).
# Step 3: Modify VGG16 model to output a feature vector
vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
for layer in vgg.layers[:-2]:
    layer.trainable = False

# Modify the output layer to match the number of classes (categories + 1 for background)
x = vgg.get_layer('fc2').output
x = Dense(len(categories) + 1, activation='softmax')(x)
model = Model(vgg.input, x)
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("vgg inputed")


# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# print("argumentation in process")

X_train = X_new
Y_train = Y_new

print("Done with setup train datas")
print("Length of X_train:", len(X_train))
print("Length of Y_train:", len(Y_train))
print("Length of X_test:", len(X_test))
print("Length of Y_test:", len(Y_test))

X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1 
)

print("argumentation in process")




train_generator = datagen.flow(X_train, Y_train, batch_size=16, subset='training')
validation_generator = datagen.flow(X_train, Y_train, batch_size=16, subset='validation')

print("Length of X_new:", len(X_new))
print("Length of Y_new:", len(Y_new))
print("Length of train_filenames:", len(train_filenames))

# After loading and preparing the train data...


# Now `test_filenames` will be defined correctly as part of the train-test split



#model.fit(X_new, Y_new, batch_size=32, epochs=3, verbose=1, validation_split=0.05, shuffle=True)
# model.fit(
#     datagen.flow(X_train, Y_train, batch_size=16),
#     epochs=5,
#     validation_data=(X_test, Y_test),
#     verbose=1,
#     # shuffle=True
# )

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    verbose=1
)
print("Model fitted with data augmentation")

# Plotting loss and analyzing losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.savefig('chart_loss.png')
plt.show()

model.save('vgg16_detector_V3.h5')


print("Model saved")

# Step 4: Using this feature vector to train an SVM.
# x = model.get_layer('fc2').output
# Y = Dense(2)(x)
# final_model = Model(model.input, Y)
# final_model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
# final_model.summary()

# # Train SVM model
# hist_final = final_model.fit(np.array(svm_images), np.array(svm_labels),
#                              #batch_size=32, epochs=10, verbose=1,
#                              batch_size=16, epochs=2, verbose=1,
#                              shuffle=True, validation_split=0.05)
# final_model.save('my_model_weights.h5')

# Step 5: Non-maximum Suppression (NMS) to remove redundant overlapping bounding boxes
def non_max_suppression(boxes, overlapThresh):
    """Perform non-maximum suppression on bounding boxes."""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Step 6: Bounding box regression to refine the positions of the bounding boxes
# Note: For simplicity, we'll skip actual regression model training. 

# Plotting loss and analyzing losses
# history = model.fit(
#     datagen.flow(X_train, Y_train, batch_size=16),
#     epochs=5,
#     validation_data=(X_test, Y_test),
#     verbose=1
# )

# # Plotting loss and analyzing losses
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title("Model Loss")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Loss", "Validation Loss"])
# plt.savefig('chart_loss.png')
# plt.show()

# model.save('vgg16_detector.h5')

# Testing on a new image
# Testing on a new image
selected_images = []
selected_labels = []
selected_filenames = []

for category in range(len(categories)):
    indices = [i for i, label in enumerate(Y_test) if label == category]
    if indices:
        chosen_index = random.choice(indices)
        selected_images.append(X_test[chosen_index])
        selected_labels.append(Y_test[chosen_index])
        selected_filenames.append(test_filenames[chosen_index])  

# Colors for ground truth and prediction
ground_truth_color = (255, 0, 0)  # Red for Ground Truth
predicted_color = (0, 255, 0)     # Green for Prediction

# Display Ground Truth and prediction boxes for all test images
fig, axes = plt.subplots(1, len(X_test), figsize=(20, 5))
for idx, (test_image, true_label, filename) in enumerate(zip(X_test, Y_test, test_filenames)):
    # Convert the test image to RGB for visualization
    image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    imOut = image.copy()

    # Construct the annotation path based on filename and category
    category_name = categories[true_label]
    annotation_path = os.path.join(test_dir, f"{category_name}_annotations", f"{filename.split('.')[0]}.csv")
    
    # Read Ground Truth boxes
    gt_boxes = pd.read_csv(annotation_path, header=None)
    for _, row in gt_boxes.iterrows():
        values = row[0].split(" ")

        if len(values) == 4:
            x1, y1, x2, y2 = map(int, values)
            cv2.rectangle(imOut, (x1, y1), (x2, y2), ground_truth_color, 2)
        else:
            print(f"Skipping row with unexpected format: {row[0]}")

    # Use selective search to generate prediction boxes
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    boxes = []

    for e, result in enumerate(ssresults[:50]):  # Only check the first 50 proposals
        x, y, w, h = result
        roi = image[y:y + h, x:x + w]
        resized_roi = cv2.resize(roi, (224, 224))
        resized_roi = np.expand_dims(resized_roi, axis=0)
        out = model.predict(resized_roi)
        
        score = np.max(out)
        class_id = np.argmax(out)
        if score > 0.5:
            boxes.append([x, y, x + w, y + h, score, class_id])

    # Apply Non-Maximum Suppression (NMS)
    boxes = np.array(boxes)
    nms_boxes = non_max_suppression(boxes, overlapThresh=0.3)

    # Draw prediction boxes
    for box in nms_boxes:
        x1, y1, x2, y2, score, class_id = box
        label_name = categories[class_id] if class_id < len(categories) else "Background"
        cv2.rectangle(imOut, (x1, y1), (x2, y2), predicted_color, 2)
        cv2.putText(imOut, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, predicted_color, 2)

    # Display the image with Ground Truth and Prediction boxes
    axes[idx].imshow(imOut)
    axes[idx].set_title(f"Ground Truth: {categories[true_label]}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

