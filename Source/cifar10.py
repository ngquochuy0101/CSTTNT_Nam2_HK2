import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

# Tải dữ liệu CIFAR-10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Lớp động vật trong CIFAR-10
animal_classes = [2, 3, 4, 5, 6, 7]
classes = ['chim', 'mèo', 'hưu', 'chó', 'ếch', 'ngựa']

# Lọc dữ liệu để chỉ giữ lại các mẫu động vật
train_mask = np.isin(Y_train, animal_classes).flatten()
test_mask = np.isin(Y_test, animal_classes).flatten()
X_train = X_train[train_mask]
Y_train = Y_train[train_mask]
X_test = X_test[test_mask]
Y_test = Y_test[test_mask]

# Kiểm tra kích thước của dữ liệu sau khi lọc
print("Kích thước dữ liệu huấn luyện:", X_train.shape)
print("Kích thước dữ liệu kiểm tra:", X_test.shape)

# Kiểm tra số lượng lớp
num_classes = len(np.unique(Y_train))
print("Số lượng lớp:", num_classes)
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

# fully connected
model_train = models.Sequential([
    Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Dropout(0.15),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(6000, activation='relu'),
    Dense(600, activation='relu'),
    Dense(60, activation='relu'),

    Dense(8, activation='softmax')
])
model_train.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

epochs = 100  # Define the number of epochs

history = model_train.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_test, Y_test))
model_train.save('model-cifar10test.h5')

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()