import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

A = 0.2
B = -20
C = 500

p_x1 = np.random.rand(10000) * 500
p_x2 = np.random.rand(10000) * 500
p_x_trigger = A * p_x1 ** 2 + B * p_x1 + C

p_x = np.zeros((10000, 2))
p_x[:, 0] = p_x1
p_x[:, 1] = p_x2
p_y = (p_x2 > p_x_trigger) * 1

val_x1 = np.random.rand(1000) * 500
val_x2 = np.random.rand(1000) * 500
val_x_trigger = A * val_x1 ** 2 + B * val_x1 + C

val_x = np.zeros((1000, 2))
val_x[:, 0] = val_x1
val_x[:, 1] = val_x2
val_y = (val_x2 > val_x_trigger) * 1

test_x1 = np.random.rand(10000) * 500
test_x2 = np.random.rand(10000) * 500
test_x_trigger = A * test_x1 ** 2 + B * test_x1 + C

test_x = np.zeros((10000, 2))
test_x[:, 0] = test_x1
test_x[:, 1] = test_x2
test_y = (test_x2 > test_x_trigger) * 1

p_y = np_utils.to_categorical(p_y, 2)
print(p_y)
val_y = np_utils.to_categorical(val_y, 2)
test_y = np_utils.to_categorical(test_y, 2)

model = Sequential()

model.add(Dense(2, activation='relu', input_shape=(2,)))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))

# model.add(Dropout(0.6))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit()