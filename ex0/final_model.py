import matplotlib.pyplot as plt
import seaborn as sns
import json

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
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers

from helpers import performance_eval, plot_samples, training_eval, save_summary, save_json, load_json, random_init


def create_lenet():
	# Create LeNet model
	input_shape = (28, 28, 1)
	model = None
	model = Sequential()

	# C1 Convolutional Layer
	model.add(Conv2D(6, kernel_size=(5, 5), strides=(1,1), activation='relu', input_shape=input_shape, padding='same'))

	# S2 Pooling Layer
	model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))

	# C3 Convolutional Layer
	model.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'))

	# S4 Pooling layer
	model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

	# C5 Fully connected conv layer

	model.add(Conv2D(120, kernel_size=(5,5), strides=(1,1), activation='relu', padding='valid'))

	# Flatten the CNN
	model.add(Flatten())

	# FC6 Fully connected layer

	model.add(Dense(84, activation='relu', activity_regularizer=regularizers.l2(0.0001)))

	# Output layer

	model.add(Dense(10, activation='softmax'))

	# compile model
	model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.9, nesterov=True), metrics=['accuracy'])

	return model

# Same random each run
random_init(42)

# Import MNIST dataset and labels from Keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Use seaborn for plots
sns.set()
sns.set_style("dark")

# Convert data to floating point and normalize to range 0-1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Reshape data for input to Conv2D layer
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Convert class labels to categorical data/one-hot encoding
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


model = create_lenet()

# Train model and evaluate training
results = model.fit(X_train, y_train, epochs=10, batch_size=64)
#training_eval(results, 'final')

# Predict and evaluate performance
y_fit = model.predict(X_test, batch_size=128)
performance_eval('final', y_fit.argmax(axis=1), y_test.argmax(axis=1))

save_json(model, 'final')
model.save_weights('models/final_weights.h5')

# Plot the problems
mis_index = np.where(y_fit.argmax(axis=1) != y_test.argmax(axis=1))
misclassifieds = X_test[mis_index]
predicted_labels = y_fit.argmax(axis=1)[mis_index]
target_labels = y_test.argmax(axis=1)[mis_index]
print('MNIST misclassifieds - predicted labels')
print(np.resize(predicted_labels, 8*12).reshape((8,12)))
print('\nMNIST misclassifieds - target labels')
print(np.resize(target_labels, 8*12).reshape((8,12)))

plot_samples(misclassifieds.reshape(94,28,28), title='MNIST_misclassifieds', width=8, height=12)