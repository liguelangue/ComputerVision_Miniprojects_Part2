# Import all necessary packages
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import argparse
import glob
import joblib

# Set up command line argument parsing for dataset selection
# The default dataset is 'manul_collect' if no dataset is specified
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="manul_collect",
                help="path to directory containing the dataset")
args = vars(ap.parse_args())

# Load the image dataset from the specified directory
print("[INFO] loading images...")
imagePaths = glob.glob(os.path.join(args["dataset"], "**", "*.*"), recursive=True)
print(f"Found {len(imagePaths)} images in dataset.")
data = []
labels = []

# Load and preprocess the images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255.0  # Normalize pixel values to the range [0, 1]
    data.append(image)

    # Extract label from the directory name
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Encode the labels using one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# Save the LabelBinarizer for future use
joblib.dump(lb, "saved_model/label_binarizer.pkl")

# Split the data into training, validation, and test sets
(trainValX, testX, trainValY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.10, random_state=42)
(trainX, valX, trainY, valY) = train_test_split(trainValX, trainValY, test_size=0.1667, random_state=42)

# Define data augmentation to generate more data from the existing dataset
aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

# Define the CNN architecture
model = Sequential()
# First convolutional layer (8 filters), input shape (256, 256, 3), ReLU activation
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(256, 256, 3)))
model.add(Activation("relu"))
# First max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer (16 filters), ReLU activation
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
# Second max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Third convolutional layer (32 filters), ReLU activation
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
# Third max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Fourth convolutional layer (64 filters), ReLU activation
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
# Fourth max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Fifth convolutional layer (128 filters), ReLU activation
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
# Fifth max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Sixth convolutional layer (256 filters), ReLU activation
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
# Sixth max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Seventh convolutional layer (512 filters), ReLU activation
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
# Seventh max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Eighth convolutional layer (1024 filters), ReLU activation
model.add(Conv2D(1024, (3, 3), padding="same"))
model.add(Activation("relu"))
# Eighth max pooling layer (2x2 pool size, stride 2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Flatten the feature maps for the fully connected layer
model.add(Flatten())
# Fully connected layer for classification, output size matches the number of classes
model.add(Dense(len(lb.classes_)))
model.add(Activation("softmax"))

# Compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model with data augmentation
print("[INFO] training network...")
valH = model.fit(trainX, trainY,
                 validation_data=(valX, valY),
                 epochs=50,
                 batch_size=32)

# Continue training the model for additional epochs with data augmentation
H = model.fit(aug.flow(trainX, trainY, batch_size=32),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // 32,
              epochs=150,
              initial_epoch=50)

# Save the model
print("[INFO] saving model...")
model.save("saved_model/model_v1.h5")  # Save as HDF5 format

# Evaluate the network on the test set
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Display 3 sample test images with predictions
sample_indices = np.random.choice(len(testX), 6, replace=False)
sample_images = testX[sample_indices]
sample_predictions = predictions[sample_indices]
sample_labels = testY[sample_indices]

for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(cv2.cvtColor(sample_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f"Pred: {lb.classes_[sample_predictions[i].argmax()]}\nTrue: {lb.classes_[sample_labels[i].argmax()]}")  
    plt.axis('off')

plt.show()
