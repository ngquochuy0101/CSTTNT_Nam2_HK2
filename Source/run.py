import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

model = models.load_model('D:\DoAn_CSTTNT\Source\model-cifar10.h5')

# Define the classes and animal classes
classes = ['chim', 'mèo', 'hưu', 'chó', 'ếch', 'ngựa']
animal_classes = [2, 3, 4, 5, 6, 7]

# Load CIFAR-10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Filter the data to keep only animal samples
train_mask = np.isin(Y_train, animal_classes).flatten()
test_mask = np.isin(Y_test, animal_classes).flatten()
X_train = X_train[train_mask]
Y_train = Y_train[train_mask]
X_test = X_test[test_mask]
Y_test = Y_test[test_mask]

# Plot images and predicted classes
a = 0
b = 0
for i in range(50):
    plt.subplot(10, 10, i + 1)
    prediction = model.predict(X_test[i].reshape((-1, 32, 32, 3)))
    predicted_class = np.argmax(prediction)

    print(predicted_class)
    a+=1
    if predicted_class == Y_test[i][0]:
      b += 1
    plt.imshow(X_test[i])
    plt.title(classes[predicted_class-2])
    plt.axis('off')
print(b / a)
plt.show()
