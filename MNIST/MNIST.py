import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

# 載入資料集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 將圖片轉為 60000 * 784 的二維向量
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32')

# 將灰階歸一化，讓數值介於 0~1 之間
x_train = x_train / 255.0 
x_test = x_test / 255.0

# 將 Lable 轉為位元，例如數字 7 => 0000001000
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 建立簡單線性模型 DNN
model = Sequential()

# 新增輸入層，輸入為 28 * 28 的向量，有 500 個變數的輸出，使用 ReLU 函數
model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))

# 新增隱藏層，有 500 個變數的輸出，使用 ReLU 函數
model.add(Dense(units=500, activation='relu'))

# 新增輸出層，有 10 個變數的輸出，使用 softmax 函數
model.add(Dense(units=10, activation='softmax'))

# 編譯模型，選擇損失函數 crossentropy，優化方法 adam，成效衡量方式 accuracy
model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

# 訓練模型，batch_size 為批次樣本數，epochs 為所有批次的最大訓練週期
model.fit(x_train, y_train, batch_size=100, epochs=20)

# 顯示訓練成果
result_train = model.evaluate(x_train, y_train)
result_test = model.evaluate(x_test, y_test)

# 印出正確率
print('Train Acc:', result_train[1])
print('Test Acc:', result_test[1])