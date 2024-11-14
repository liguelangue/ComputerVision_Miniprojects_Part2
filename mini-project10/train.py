import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import random

# Image and mask paths

image_path = "./weizmann_horse_db/horse/*.png"
mask_path = "./weizmann_horse_db/mask/*.png"

# Preview image and mask

def preview_image_and_mask(image_paths, mask_paths, target_size=(256, 256)):
    idx = random.randint(0, len(image_paths) - 1)
    image_path = image_paths[idx]
    mask_path = mask_paths[idx]

    # Load image and add grayscale channel
    image_rgb = plt.imread(image_path).astype(np.float32)
    if image_rgb.max() > 1.0:
        image_rgb /= 255.0
    image_rgb = cv2.resize(image_rgb, target_size)

    # Add grayscale channel
    image_gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    image = np.dstack((image_rgb, image_gray))

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if mask.max() > 1.0:
        mask /= 255.0
    mask = cv2.resize(mask, target_size)
    mask = (mask > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow((image_rgb * 255).astype(np.uint8))
    axes[0].set_title('Preview Image (RGB)')
    axes[0].axis('off')

    axes[1].imshow((mask * 255).astype(np.uint8), cmap='gray')
    axes[1].set_title('Preview Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Load and prepare data
def load_and_prepare_data(image_paths, mask_paths, target_size=(256, 256)):
    image_list, mask_list = [], []
    for image_path, mask_path in zip(image_paths, mask_paths):
        # Load image and normalize
        image_rgb = plt.imread(image_path).astype(np.float32)
        if image_rgb.max() > 1.0:
            image_rgb /= 255.0
        image_rgb = cv2.resize(image_rgb, target_size)

        # Add grayscale channel
        image_gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        image = np.dstack((image_rgb, image_gray))

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0
        mask = cv2.resize(mask, target_size)
        mask = (mask > 0.5).astype(np.float32)

        image_list.append(image)
        mask_list.append(mask)

    return np.array(image_list), np.array(mask_list)

# Create Attention U-Net model
def create_attention_unet(input_shape=(256, 256, 4)):
    inputs = tf.keras.layers.Input(input_shape)

    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        return x

    def attention_block(skip, gating, filters):
        theta_x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(skip)
        phi_g = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(gating)
        add = tf.keras.layers.add([theta_x, phi_g])
        act = tf.keras.layers.Activation('relu')(add)
        psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(act)
        return tf.keras.layers.multiply([skip, psi])

    # Encoder part
    enc1 = conv_block(inputs, 32)
    enc2 = tf.keras.layers.MaxPooling2D((2, 2))(enc1)
    enc3 = conv_block(enc2, 64)
    enc4 = tf.keras.layers.MaxPooling2D((2, 2))(enc3)
    enc5 = conv_block(enc4, 128)
    enc6 = tf.keras.layers.MaxPooling2D((2, 2))(enc5)
    enc7 = conv_block(enc6, 256)
    enc8 = tf.keras.layers.MaxPooling2D((2, 2))(enc7)
    enc9 = conv_block(enc8, 512)

    # Decoder part + Attention mechanism
    def upconv_block(x, skip, filters):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        attention = attention_block(skip, x, filters // 2)
        x = tf.keras.layers.Concatenate()([x, attention])
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        return x

    # Decoder with skip connections to corresponding encoder layers
    dec1 = upconv_block(enc9, enc7, 256)
    dec2 = upconv_block(dec1, enc5, 128)
    dec3 = upconv_block(dec2, enc3, 64)
    dec4 = upconv_block(dec3, enc1, 32)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(dec4)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Load paths and data
image_paths = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_paths = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])
images, masks = load_and_prepare_data(image_paths, mask_paths)
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=23)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1):
    return 1 - dice_coef(y_true, y_pred, smooth)

def combined_dice_bce_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice + bce

# Create and compile model
model = create_attention_unet(input_shape=(256, 256, 4))
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss=combined_dice_bce_loss,
              metrics=['accuracy', dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)])

# Define model saving callbacks
model_checkpoint_best = ModelCheckpoint('unet_horse_best_v2.h5', monitor='val_loss', save_best_only=True, mode='min')
model_checkpoint_last = ModelCheckpoint('unet_horse_final_v2.h5', save_best_only=False)

# Train model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test), 
                    callbacks=[reduce_lr, model_checkpoint_best, model_checkpoint_last])

# Display prediction results
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

def display_predictions(model, X_test, y_test):
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for i in range(3):
        rand_idx = random.randint(0, len(X_test) - 1)
        augmented_img = X_test[rand_idx]
        original_mask = y_test[rand_idx]

        # Predict mask
        predicted_mask = model.predict(np.expand_dims(augmented_img, axis=0))
        predicted_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(int)

        # Calculate IoU
        iou_score = calculate_iou(original_mask.squeeze(), predicted_mask)

        # Visualization
        augmented_img_vis = (augmented_img[:, :, :3] * 255).astype(np.uint8)
        original_mask_vis = (original_mask * 255).astype(np.uint8)
        predicted_mask_vis = (predicted_mask * 255).astype(np.uint8)

        axes[i, 0].imshow(augmented_img_vis)
        axes[i, 0].set_title('Test Image')

        axes[i, 1].imshow(original_mask_vis, cmap='gray')
        axes[i, 1].set_title('Original Mask')

        axes[i, 2].imshow(predicted_mask_vis, cmap='gray')
        axes[i, 2].set_title(f'Predicted Mask\nIoU: {iou_score:.2f}')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

display_predictions(model, X_test, y_test)
