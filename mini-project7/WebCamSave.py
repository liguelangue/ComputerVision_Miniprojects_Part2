# USAGE: python WebCamSave.py -f video_file_name -o out_video.avi

# import the necessary packages
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os
import glob
import argparse

class LiveClassifier:

    def load_data(self):
        # 定义参数解析器
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", type=str, help="路径到包含图像数据的文件夹")
        ap.add_argument("--train_file", type=str, help="路径到训练数据的 h5 文件")
        ap.add_argument("--test_file", type=str, help="路径到测试数据的 h5 文件")
        args = vars(ap.parse_args())

        # --dataset
        if args["dataset"]:
            print("[INFO] loading dataset...")
            imagePaths = glob.glob(os.path.join(args["dataset"], "**", "*.jpg"), recursive=True)
            data = []
            labels = []

            # Load and preprocess the images
            for imagePath in imagePaths:
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (128, 128))
                image = image.astype("float32") / 255.0
                data.append(image)

                label = imagePath.split(os.path.sep)[-2]  # 从路径中提取类别名
                labels.append(label)

            # Encode the labels
            lb = LabelBinarizer()
            labels = lb.fit_transform(labels)
            return np.array(data), labels, None, None, lb  # 返回数据和标签，其他为 None

        # --train_file and --test_file (h5 file)
        elif args["train_file"] and args["test_file"]:
            print("[INFO] loading h5 dataset...")
            # 加载训练数据
            with h5py.File(args["train_file"], "r") as f:
                trainX = np.array(f['train_set_x'][:])  # 获取训练图像数据
                trainY = np.array(f['train_set_y'][:])  # 获取训练标签

            # 加载测试数据
            with h5py.File(args["test_file"], "r") as f:
                testX = np.array(f['test_set_x'][:])  # 获取测试图像数据
                testY = np.array(f['test_set_y'][:])  # 获取测试标签

            # 数据归一化处理
            trainX = trainX.astype("float32") / 255.0
            testX = testX.astype("float32") / 255.0

            # 独热编码标签
            lb = LabelBinarizer()
            trainY = lb.fit_transform(trainY)
            testY = lb.transform(testY)

            return trainX, trainY, testX, testY, lb  # 返回训练和测试数据

        else:
            raise ValueError("必须提供 --dataset 或 --train_file 和 --test_file 参数之一来加载数据")
    
    def split_data(self, trainX=None, trainY=None, testX=None, testY=None, validation_split=0.2):
        # 如果没有预分配测试数据 (比如从图片文件夹加载的情况)
        if testX is None and testY is None:
            print("[INFO] 数据未预分割，开始手动分割...")
            # 第一步：先将数据分成训练集和临时集（临时集包含验证集和测试集）
            trainX, tempX, trainY, tempY = train_test_split(np.array(trainX), np.array(trainY), test_size=0.4, random_state=42)

            # 第二步：将临时集再分割为验证集和测试集
            valX, testX, valY, testY = train_test_split(tempX, tempY, test_size=0.5, random_state=42)

            return trainX, trainY, valX, valY, testX, testY

        else:
            print("[INFO] 数据已预分割，开始划分验证集...")
            # 如果已经有训练集和测试集，比如使用了 h5 文件，只需划分训练集为训练集和验证集
            trainX, valX, trainY, valY = train_test_split(np.array(trainX), np.array(trainY), test_size=validation_split, random_state=42)

            # 直接返回训练集、验证集和测试集
            return trainX, trainY, valX, valY, testX, testY
    
    def cnn(self, lb):
        model = Sequential()

        # 根据你的数据集，调整输入形状
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(128, 128, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Flatten())  # 展平
        model.summary()
        # # 全连接层
        # model.add(Dense(8192))  # 修改为与前面层匹配的大小
        # model.add(Activation("relu"))

        # 输出层，类别数量为 len(lb.classes_)
        model.add(Dense(len(lb.classes_)))
        model.add(Activation("softmax"))

        return model

    def model_compile(self, model):
        print("[INFO] model compiling...")
        opt = Adam(learning_rate=1e-3)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    # 创建回调函数
    def create_callbacks(self):
        callbacks = []

            # 动态学习率调度函数
        def lr_schedule(epoch):
            initial_lr = 1e-3
            return initial_lr * (1 / (1 + 0.01 * epoch))
        
        # 动态学习率调度
        callbacks.append(LearningRateScheduler(lr_schedule))

        # 提前停止训练
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",  # 监控验证集损失
            patience=10,  # 如果验证集损失连续 10 个 epoch 没有改善，则停止训练
            verbose=1
        )
        callbacks.append(early_stopping_callback)

        # 当验证集损失停滞时自动降低学习率
        reduce_lr_callback = ReduceLROnPlateau(
            monitor="val_loss",  # 监控验证集损失
            factor=0.1,  # 每次降低学习率的因子
            patience=5,  # 如果验证集损失连续 5 个 epoch 没有改善，则降低学习率
            verbose=1
        )
        callbacks.append(reduce_lr_callback)

        return callbacks
    
    def train_model(self, model, trainX, trainY, valX, valY, callbacks):
        print("[INFO] training network...")

        # # 使用 ImageDataGenerator 进行数据增强
        # datagen = ImageDataGenerator(
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     zoom_range=0.5,
        #     horizontal_flip=True,
        #     fill_mode='nearest'
        # )

        # # 对训练数据应用数据增强
        # datagen.fit(trainX)
        
        model.fit(trainX, 
                  trainY, 
                  validation_data=(valX, valY), 
                  epochs=100, 
                  batch_size=16,
                  callbacks=callbacks)
    
    def display_sample_predictions(self, model, testX, testY, lb):
        # Evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=32)

        # 将 lb.classes_ 转换为字符串
        target_names = lb.classes_.astype(str)

        # 使用分类报告显示结果
        print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))

        # 随机选择 3 个测试集中的样本
        sample_indices = np.random.choice(len(testX), 3, replace=False)
        sample_images = testX[sample_indices]
        sample_predictions = predictions[sample_indices]
        sample_labels = testY[sample_indices]

        # 创建一个 1 行 3 列的网格来显示图像
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 行 3 列的子图

        for i, (image, pred, label) in enumerate(zip(sample_images, sample_predictions, sample_labels)):
            # 在第 i 个子图中显示图像
            axs[i].imshow(image)

            # 获取预测的类索引和真实类索引
            predicted_label = np.argmax(pred)
            true_label = np.argmax(label)

            # 在标题中显示预测结果和真实标签
            axs[i].set_title(f"Predicted: {target_names[predicted_label]}\nTrue: {target_names[true_label]}")

            # 去掉坐标轴
            axs[i].axis('off')

        # 显示图像
        plt.tight_layout()
        plt.show()
            
    def live_detection(self, model, lb):
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

        # loop over the frames from the video stream
        while True:
            # grab the frame from video stream
            ret, frame = vs.read()
            if not ret:
                break

            # Add your code HERE:
            # resize and normalize the frames
            resized_frame = cv2.resize(frame, (32, 32))
            normalized_frame = resized_frame.astype("float32") / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)

            # prediction from CNN model
            predictions = model.predict(input_frame)
            predicted_class = np.argmax(predictions[0])
            label = lb.classes_[predicted_class]
            
            cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write the frame to the output video file
            if args.out:
                out.write(frame)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # Release the video capture object
        vs.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier = LiveClassifier()

    # 加载数据
    trainX, trainY, testX, testY, lb = classifier.load_data()

    # 如果使用图片文件夹，则还需要进一步划分训练和验证集
    if testX is None and testY is None:
        trainX, trainY, valX, valY, testX, testY = classifier.split_data(trainX, trainY)
    else:
        # 已经有测试集的情况
        trainX, trainY, valX, valY, testX, testY = classifier.split_data(trainX, trainY, testX, testY)

    # 构建和编译模型
    model = classifier.cnn(lb)
    model = classifier.model_compile(model)

    # 创建回调函数
    callbacks = classifier.create_callbacks()

    # 训练模型，传入回调函数
    classifier.train_model(model, trainX, trainY, valX, valY, callbacks)

    # 展示3个测试样本的预测结果
    classifier.display_sample_predictions(model, testX, testY, lb)

    # 启动实时检测
    classifier.live_detection(model, lb)