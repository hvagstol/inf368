# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:50:00 2019

@author: hvagstol@gmail.com
"""

import numpy as np
import seaborn as sns
import h5py

from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers

from helpers import performance_eval, plot_samples, training_eval, save_summary, save_json, load_json, random_init

# Same random each run
random_init(42)

# Import MNIST dataset and labels from Keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Use seaborn for plots
sns.set()
sns.set_style("dark")

# Display shape of data on terminal
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)

# Plot a few examples
sample_indices = [1, 2, 4, 8, 16 , 32, 65, 1329,4000]
plot_samples(X_train[sample_indices],'MNIST samples')

# Convert data to floating point and normalize to range 0-1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


# Convert class labels to categorical data/one-hot encoding
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


# Reshape data for input to Conv2D layer
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Create LeNet model
input_shape = (28, 28, 1)
model = Sequential()

# C1 Convolutional Layer
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1,1), activation='tanh', input_shape=input_shape, padding='same'))

# S2 Pooling Layer
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))

# C3 Convolutional Layer
model.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), padding='valid'))

# S4 Pooling layer
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# C5 Fully connected conv layer

model.add(Conv2D(120, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))

# Flatten the CNN
model.add(Flatten())

# FC6 Fully connected layer

model.add(Dense(84, activation='tanh'))

# Output layer

model.add(Dense(10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(), metrics=['accuracy'])


# Write the summary to file 
save_summary(model, 'lenet')

# Train model and evaluate training
results = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=1/12)
training_eval(results, 'lenet')

# Predict and evaluate performance
y_fit = model.predict(X_test, batch_size=128)
performance_eval('lenet', y_fit.argmax(axis=1), y_test.argmax(axis=1))

save_json(model, 'lenet')
model.save_weights('models/lenet_weights.h5')