import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# 定义 Dice 系数
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


# 加载训练好的模型
model = tf.keras.models.load_model(r'D:\NEU\CS5330\mini_proj_10\unet_horse_final_v2.h5', 
                                   custom_objects={'dice_coef': dice_coef, 'combined_dice_bce_loss': combined_dice_bce_loss})

# 加载单张图像并处理
def load_single_image(image_path, target_size=(256, 256)):
    image_rgb = plt.imread(image_path).astype(np.float32)
    if image_rgb.max() > 1.0:
        image_rgb /= 255.0  # 归一化至 [0, 1]
    image_rgb = cv2.resize(image_rgb, target_size)

    # 添加灰度通道
    image_gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    image = np.dstack((image_rgb, image_gray))
    
    # 检查通道数和归一化情况
    print("Image shape (should be 4 channels):", image.shape)
    print("Image pixel range:", image.min(), "-", image.max())
    
    return image





# 预测并显示结果
def test_single_image_no_mask(model, image_path):
    image = load_single_image(image_path)

    predicted_mask_raw = model.predict(np.expand_dims(image, axis=0))[0, :, :, 0]
    print("Predicted mask raw values:", np.unique(predicted_mask_raw))
    
    # 进行预测
    predicted_mask = model.predict(np.expand_dims(image, axis=0))
    predicted_mask = (predicted_mask[0, :, :, 0] > 0.2).astype(int)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow((image[:, :, :3] * 255).astype(np.uint8))
    axes[0].set_title('Input Image')
    
    axes[1].imshow((predicted_mask * 255).astype(np.uint8), cmap='gray')
    axes[1].set_title('Predicted Mask')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用单张图片路径测试模型
test_single_image_no_mask(model, r"D:\NEU\CS5330\mini_proj_10\test_img\9.png")
