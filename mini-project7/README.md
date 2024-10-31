# CS5330_F24_Group8_Mini_Project_7
# Project Members
Anning Tian, Pingyi Xu, Qinhao Zhang, Xinmeng Wu

# Setup Instructions
Download the zip file or use GitHub Desktop to clone the file folder.

WebCamSave.py: This file is the model training script, which includes methods for training the model.

live_test.py: This file is used to apply the trained model in a live setting using a webcam. It captures live video streams and uses the trained model to make predictions on the captured frames, demonstrating the real-time performance of the model.

The saved_model folder contains the following files:
- model_v1.h5: This is the trained model file, which stores the network weights and structure to be used for making predictions on new data.
- label_binarizer.pkl: This file is a trained label binarizer used to convert classification labels into a format suitable for the machine learning model. Specifically, it converts category labels into one-hot encoding and can also convert one-hot encoded labels back to their original class names. During the prediction process, the model outputs numerical arrays, and label_binarizer.pkl helps map these arrays back to their corresponding class names, making it easier to understand and analyze the model's predictions.

The data_preprocessing.py script is used for renaming images based on the folder in which each image is located. It iterates over each subfolder in a specified parent folder, renaming each image in the subfolder by appending the subfolder name as a prefix, followed by a unique counter to avoid duplicate names. The script ensures that each image has a distinct name, even if there are name conflicts.

The Model_Result.jpg contains a screenshot of the model's prediction results. In the screenshot, the images are shown with both their predicted labels and true labels. Below the images, a classification report table is provided, displaying precision, recall, F1-score for each category, and overall metrics like accuracy. This information helps in evaluating the model's performance across different categories and its overall effectiveness.

# usage guide
Run the LiveCam file:
```
python live_test.py
```

In the upper left corner, it shows the FPS and the name of the detected object.

# Dataset Information
https://drive.google.com/drive/folders/1lepJM4a6fBqTWQajXSQDB5o8iK8lRQn8?usp=sharing

## Details about the dataset
bottle: 102

remote: 336

phone: 362

TV: 222

## Data Preprocessing

Image Cropping and Adjustment: A large number of images were processed by cropping them to ensure the target object occupies the majority of the image area. This step helps center and focus on the object of interest for clearer feature extraction.

Renaming, Sorting, and Classification: The collected images were systematically renamed, sorted, and categorized in batches, facilitating better organization and efficient access during further processing and training.

# CNN Model
## CNN - layers
Input Layer: Images are resized to 256×256 with 3 color channels.

Convolutional Layers: The model has eight convolutional layers, each followed by the ReLU activation function. These layers extract increasingly complex features as we progress:

- First Layer: Conv2D(8, (3, 3), padding="same") with ReLU activation.

- Second Layer: Conv2D(16, (3, 3), padding="same") with ReLU activation.

- Third Layer: Conv2D(32, (3, 3), padding="same") with ReLU activation.

- Fourth Layer: Conv2D(64, (3, 3), padding="same") with ReLU activation.

- Fifth Layer: Conv2D(128, (3, 3), padding="same") with ReLU activation.

- Sixth Layer: Conv2D(256, (3, 3), padding="same") with ReLU activation.

- Seventh Layer: Conv2D(512, (3, 3), padding="same") with ReLU activation.

- Eighth Layer: Conv2D(1024, (3, 3), padding="same") with ReLU activation.

Pooling Layers: Each convolutional layer (except the output layers) is followed by a MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) layer, reducing the spatial dimensions by half, which reduces the computational cost and extracts dominant features.

Flatten Layer: Converts the final pooled feature maps into a 1D vector to prepare for fully connected layers.

Dense Layer: Dense(len(lb.classes_)), which outputs a number of neurons equal to the number of classes in the dataset, followed by a softmax activation for multi-class classification.
## CNN - activation functions
ReLU (Rectified Linear Unit) is used in all convolutional layers to introduce non-linearity, allowing the model to learn more complex patterns.

Softmax in the output layer provides a probability distribution across classes.
## CNN - optimization techniques
Data Augmentation: ImageDataGenerator with various augmentations (rotation, width/height shifts, shearing, zooming, and horizontal flipping) is applied to prevent overfitting and improve generalization, especially useful for smaller datasets.

Adam Optimizer: The model uses the Adam optimizer with a learning rate of 10^(-3) and a decay rate of 10^(-3)/50 over 50 epochs, which helps in adaptive learning.

Loss Function: Categorical cross-entropy is used as the loss function, suitable for multi-class classification.

# Model Evaluation
## Accuracy and loss during training
Accuracy: During training, the model’s accuracy improved consistently, which means it was learning from the data well. The validation accuracy plateaued around 87%, suggesting that the model generalizes reasonably well on unseen data. However, fluctuations in the later epochs indicate some level of overfitting, where the model may be learning noise from the training set rather than generalizable patterns.

Loss: The training loss decreased steadily and reached a very low level, indicating that the model learned the training data well. Validation loss initially decreased but started fluctuating in the later epochs, which suggests potential overfitting.

## Model's performance metrics
Phone: Precision of 0.80, recall of 0.85, and F1-score of 0.82. This indicates moderate performance, with a balance between precision and recall, though there may still be a few misclassifications.

TV: Precision of 0.80, recall of 0.96, and F1-score of 0.87. This shows strong recall, meaning the model successfully identifies almost all TV instances, though precision is slightly lower.

Bottles: Precision of 0.92, recall of 0.81, and F1-score of 0.86, reflecting good model performance on this category, with high precision and a slight drop in recall.

Remote: Precision of 0.89, recall of 0.77, and F1-score of 0.83, indicating reliable detection, but with a relatively lower recall, meaning some "remote" instances may be missed.

## Challenges
Overfitting: The model reached high accuracy on the training set, but validation accuracy plateaued and validation loss fluctuated. Adding regularization techniques, like dropout, or using early stopping, could help mitigate overfitting.

Class Imbalance: The model had more difficulty with some categories. This could be due to class imbalance or less distinguishable features for this category. More data or targeted augmentation might help improve its performance.

Performance on Small Dataset: Given the good performance metrics, it seems the data preprocessing (cropping, renaming, and organizing) and data augmentation contributed positively, though further dataset expansion could enhance the model’s robustness.

# The link to the video demonstrating
https://drive.google.com/file/d/16cShBRE_qC22WpVMg75ucgmQRWxBtn42/view?usp=sharing

