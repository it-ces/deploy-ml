import numpy as np
import matplotlib.pyplot as plt
from PIL  import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = cifar10.load_data() # images of differnts categories, animales, etc....
X_train  = X_train/255
X_test  = X_test/255 # standardize is better for neural networks



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model = Sequential([ Flatten(input_shape =(32,32, 3)), #rgb
    Dense(500, activation = 'relu'),
    Dense(500, activation = 'relu'),
    Dense(10, activation = 'softmax') # 10 categoricals all probabalties must sum up 1.

])

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
model.fit(X_train, y_train, batch_size = 64, epochs = 30, validation_data=(X_test, y_test))
model.save('cifar10_model.keras')
