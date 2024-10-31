# import all packages we need
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

# set up user use for img dataset select
# -- default means set the default dataset if no dataset select
# -d or -dataset are both okay to use for select the ds
# -help can help user understand what here is doing
# vars() turn the ap.parse_args() to a dic format
# args finally become a dic
# args = {'dataset': '/path/to/dataset'} -> this will be how it looks like
# it is a dic format and stores the name of ds and path to it
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="manul_collect",
                help="path to directory containing the need dataset")
args = vars(ap.parse_args())

# first print we status loding the img
# get the path of imge from args["dataset"] == '/path/to/dataset', and use the "**/*.jpg" to select all img under that directory

print("[INFO] loading images...")
imagePaths = glob.glob(os.path.join(args["dataset"], "**", "*.*"), recursive=True)
print(f"Found {len(imagePaths)} images in dataset.")
data = []
labels = []

# Load and preprocess the images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    # turn image pixel to float and turn pixel from range of 0-255 to 0-1
    image = image.astype("float32") / 255.0
    data.append(image)

    # set up the label for each img, here we do this bc we are assuming that the name of img it's its lable
    # for ex: /path/to/dataset/cat/image1.jpg, here label stores cat
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Encode the labels
lb = LabelBinarizer() # one-hot encode the label, for ex, we have total 3 labels, so eahc will be: [0,0,1] or [0,1,0] or [1,0,0]
labels = lb.fit_transform(labels) # transform the lb to labels, let the label in labels become: labels = [[0,1,0], [0,0,1], [1,0,0]]
# 保存 LabelBinarizer 到文件
joblib.dump(lb, r"D:\NEU\CS5330\mini_proj_7\saved_model\label_binarizer.pkl")

# Perform a training and testing split
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

# method1: Define data augmentation
# this is incase we dont have enough data(imgs), we can use rotate, zoom, flip, etc.. to add more images for train and test
# the modification is randomly generate each time along with input to apply ranmdo effect/effects to it
# this is good for small dataset and also good for generalize the model ability
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
# first cov layer, turn 1 input to 8 feature imgs, the core bracket for feature extraction is 3 X 3 bracket.
# define the input img shape
# set padding to same so we have input size of img == feature img size
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(128, 128, 3)))
# set activation to relu for first cov layer
# help model learn non-linaer features
model.add(Activation("relu"))
# add first pool layer, pool bracket(window) is 2 x 2 bracket --> pool_size=(2, 2)
# set the step by using strides=(2, 2) means we are step2 each time
# by combine the pool an dstrides, we can turn 32 x 32 img to 16 x 16
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# second cov layer, remain the same size after first pool
# extract 16 features this time for each of 8(from after first cov)
# activatiuo still as relu
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
# doing pooling again, now cut each image size half again from second cov layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# add third cov layer, and now we have 32 feature img exract from eahc 16 of above after second pooling
# activation still set to relu
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
# cut the after cov layer image size to half again, only remain max value
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# turn the each feature after last pool to a flatten like
# for ex: afetr last pool we have: [batch size(numbers of featurns), 3, 3, 32(numbers of channel, number of channel means the number of features at last cov layer)]
# for ex, here we have last cov with layer of 32, so numbers of featurs will be 32
# bacth size just as batch size
# so here after flatter, for example we are having batch size of 10
# so, it will be form: [10, 3, 3, 32] -> [10, 311]

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


# model.add(Conv2D(1024, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())


# 这里的 Dense(len(lb.classes_)) 是全连接层，输出神经元的数量等于要分类的类别数。
# 如果有 3 个类别，那么 len(lb.classes_) = 3
# 输出层会有 3 个神经元，每个神经元表示一个类别的预测。
model.add(Dense(len(lb.classes_)))
# softmax 激活函数用于将最后的输出转换为概率分布，适合多分类任务。
# 它会确保输出的 3 个神经元的值加起来等于 1，每个值表示属于该类别的概率。
model.add(Activation("softmax"))


# Compile the model
print("[INFO] compiling model...")
# Adam 是一种常用的优化器，
# 结合了 Momentum 和 RMSProp 的优点。它通过自适应调整学习率，
# 能在复杂的优化问题中表现出色，通常比标准的随机梯度下降（SGD）更快收敛。

# Adam 中的两个核心超参数是 学习率（learning rate） 和 学习率衰减（decay）。
# 这是优化器的初始学习率，即每次权重更新的步幅大小。1e-3 等于 0.001。
# 学习率越大，模型更新参数的速度越快，
# # 但可能会导致训练不稳定或错过最优解。学习率太小，
# # 则可能导致训练速度过慢，模型需要较长时间才能收敛。

# 这里定义了学习率的衰减值，表示学习率在每个更新步骤中逐步降低。衰减的公式是 1e-3 / 50 = 0.00002。
# 学习率衰减用于在训练过程中逐渐减少学习率，这有助于在训练初期快速逼近最优解，同时在训练后期保持稳定，避免错过局部最优。
# 随着训练轮数（epochs）增加，学习率会慢慢减小，帮助模型更精细地调整权重。
opt = Adam(learning_rate=1e-3, decay=1e-3 / 50)
# 这一行是编译模型的---核心部分---，定义了模型训练时需要用到的三个关键组件：损失函数、优化器和评估指标。
# loss="categorical_crossentropy" loss 定义了训练过程中使用的 损失函数，它衡量模型的预测结果与真实标签之间的差异。
# categorical_crossentropy 是一种用于多分类问题的损失函数，它通常用于有两个以上类别的分类任务（例如图片分类任务）。
# 交叉熵损失（cross-entropy loss） 会衡量模型输出的概率分布与真实标签分布之间的差距。模型输出的类别概率越接近真实的类别分布，损失就越小。
# 对于多分类任务，categorical_crossentropy 适用于独占类别的情况，也就是说每个样本只属于一个类别（例如：一张图片只能是猫、狗或车中的一种）。
# optimizer 定义了模型使用的优化器，这里是之前定义的 Adam 优化器。
# 优化器的作用是在每个训练步骤中，根据损失函数的值，计算出每个权重的梯度，并更新模型的权重。
# Adam 通过动态调整学习率和梯度方向来有效地优化模型参数。
# metrics 定义了模型的性能评估指标。在训练和评估过程中，Keras 会根据指定的指标评估模型的表现。
# accuracy：这里选择了 准确率（accuracy） 作为评估指标，表示分类正确的样本占总样本的比例。对于分类问题，准确率是最常用的衡量标准之一。
#  在训练过程中，Keras 会计算并输出训练集和验证集上的准确率，帮助你评估模型的表现。
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model with data augmentation
print("[INFO] training network...")

# 这段代码调用了 model.fit() 来开始训练模型。
# Keras 中的 fit() 函数负责模型的训练过程，它会根据训练数据进行前向传播、计算损失、进行梯度反向传播，并更新模型的权重，直到训练完成。
# 你提到的代码使用了数据增强并指定了训练和验证集

# 变量 H 用于存储模型训练的历史记录（history）。
# H 会包含训练过程中每个 epoch 的损失值和评估指标（如准确率）。
# 这可以用于后续的可视化（如绘制训练损失、验证准确率等）。
# 可以通过 H.history 访问训练过程中的历史数据。

# aug.flow(trainX, trainY, batch_size=32)
# aug：这是你前面定义的 数据增强生成器（ImageDataGenerator），
# 它对输入图像进行实时的数据增强，例如旋转、平移、缩放等操作。这样可以扩展训练数据集，并提升模型的泛化能力。
# trainX 和 trainY 是训练数据和对应的标签。trainX 是输入的图像数据，trainY 是对应的类别标签。
# batch_size=32：指定每次迭代时模型处理的样本数量为 32。数据增强生成器会每次从 trainX 中随机选择 32 张图像进行增强，并生成一个批次的训练数据。
# aug.flow() 会生成一个增强后的数据批次，每次迭代时都会将新的批次传入模型。
# validation_data：验证数据是用于评估模型在每个 epoch 结束时的表现，以监控模型的泛化能力和防止过拟合。
# testX 是验证集中的输入图像，testY 是对应的标签。验证集的输入数据不会进行数据增强，它们会被直接传入模型。
# 模型不会对验证集的数据进行权重更新，它只会计算损失和评估指标（例如准确率），用于评估模型在未见过的数据上的表现。
# 验证集不参与训练，只用于衡量模型的性能。如果验证集的损失持续增加而训练集的损失下降，则表明模型可能出现了过拟合。
# steps_per_epoch=len(trainX) // 32
# steps_per_epoch：每个 epoch 中的迭代次数（steps）。这是模型在每个 epoch 里训练时需要进行的前向传播和反向传播的总步数。
# len(trainX) // 32：这里表示将训练集的总样本数量 trainX 按照 batch_size=32 分割为批次。
# 每个 epoch 会迭代 len(trainX) // 32 次，即训练集中的所有样本都会被用到一次。
# 例如，如果 trainX 中有 1000 张图像，batch_size 是 32，那么 steps_per_epoch 将是 1000 // 32 = 31 步，意味着每个 epoch 需要进行 31 次权重更新。
# epochs=50：指定训练的总轮数（epochs），即模型将训练 50 轮。
# Train the model with data augmentation
print("[INFO] training network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=32),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // 32,
              epochs=150)

# Save the model
print("[INFO] saving model...")
model.save(r"D:\NEU\CS5330\mini_proj_7\saved_model/model_v1.h5")  # 保存为 HDF5 格式

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Display 3 sample test images with predictions
sample_indices = np.random.choice(len(testX), 3, replace=False)
sample_images = testX[sample_indices]
sample_predictions = predictions[sample_indices]
sample_labels = testY[sample_indices]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(sample_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f"Pred: {lb.classes_[sample_predictions[i].argmax()]}\nTrue: {lb.classes_[sample_labels[i].argmax()]}")  
    plt.axis('off')

plt.show()
