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
from keras.models import Sequential
from keras.layers import Dense

from helpers import performance_eval, plot_samples, training_eval, save_summary, save_json, random_init

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


# Reshape data for input to Dense layer
X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

# Create sequential 2-layer model
model = Sequential()
model.add(Dense(400, input_dim=28*28, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(), metrics=['accuracy'])

# Write the summary to file 
save_summary(model, 'simple')

# Train model and evaluate training
results = model.fit(X_train.reshape(-1,28*28), y_train, epochs=10, batch_size=64, validation_split=1/12)
training_eval(results, 'simple')

# Predict and evaluate performance
y_fit = model.predict(X_test, batch_size=128)
performance_eval('simple', y_fit.argmax(axis=1), y_test.argmax(axis=1))

save_json(model, 'simple')
model.save_weights('models/simple_weights.h5')