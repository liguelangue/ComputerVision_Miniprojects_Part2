import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 加载已保存的 LabelBinarizer
lb = joblib.load(r'label_binarizer.pkl')

# 加载模型
model = load_model(r"model_v1.h5")

# 加载并预处理图像
def prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # 假设模型输入是 128x128
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # 增加一个 batch 维度
    return image

# 进行预测
def predict_image(image_path, model):
    image = prepare_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

# 调用函数预测图像
image_path = r'test_image/5.jpeg'
predicted_class = predict_image(image_path, model)

# 将类别索引转换为具体类别名称
predicted_label = lb.classes_[predicted_class[0]]
print(f"Predicted label: {predicted_label}")
