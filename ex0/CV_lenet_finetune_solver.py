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

from helpers import performance_eval, plot_samples, training_eval, save_summary, save_json, load_json, random_init

def skf_cross_validate(model, X, y, title):
	"""
	wrapper function to do sklearn style stratified k fold cross validation
	on a keras model. Some code borrowed from:
	https://medium.com/@literallywords/stratified-k-fold-with-keras-e57c487b1416

	input - X, training data
			y, training data labels
	        
	"""
	print("Stratified K Fold cross Validating")
	skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
	for index, (train_index, test_index) in enumerate(skf.split(X, y)):

		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		# Convert class labels to categorical data/one-hot encoding
		y_test = to_categorical(y_test)
		y_train = to_categorical(y_train)


		print ('Training '+'lenet_finetune_'+title+'_'+str(index))
		# Train model and validate
		results = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
		with open('output/lenet_finetune_'+title+'_'+str(index)+'_history.json','w') as f:
			json.dump(results.history, f)
		model.save_weights('models/lenet_finetune_'+title+'_'+str(index)+'_weights.h5')
		    

def create_lenet(optimizer='standard'):
	# Create LeNet model
	input_shape = (28, 28, 1)
	model = None
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
	if (optimizer == 'sgd'):
		model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(), metrics=['accuracy'])
	elif (optimizer == 'momentum'):
		model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.7), metrics=['accuracy'])
	elif (optimizer == 'nesterov'):
		model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.7, nesterov=True), metrics=['accuracy'])


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
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

skf_cross_validate(create_lenet('sgd'), X_train, y_train, 'sgd')
skf_cross_validate(create_lenet('momentum'), X_train, y_train, 'momentum')
skf_cross_validate(create_lenet('nesterov'), X_train, y_train, 'nesterov')

# summarize history for validation accuracy

for version in ['sgd', 'momentum', 'nesterov']:

	all_val_acc = []
	plt.clf()
	for index in range(5):
		lenet_history = None
		with open('output/lenet_finetune_'+version+'_'+str(index)+'_history.json', 'r') as f:
			lenet_history = json.loads(f.read())

		plt.plot(lenet_history['val_acc'])
		all_val_acc.append(lenet_history['val_acc'][-1])

	plt.title('lenet stratified k fold cv accuracy, '+version+' optimizer')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['lenet-finetune-'+version+'0', 'lenet-finetune-'+version+'-1', 'lenet-finetune-'+version+'-2', 'lenet-finetune-'+version+'-3', 'lenet-finetune-'+version+'-4'], loc='upper left')
	plt.savefig('output/lenet_finetune_'+version+'_validation_accuracy.png', bbox_inches='tight')

	#print(all_val_acc)
	print(version + ' validation accuracy mean %.6f' % np.mean(all_val_acc))
	print(version + ' validation accuracy variance %.6f' % np.var(all_val_acc))

	with open('output/lenet_finetune_'+version+'_val_acc.txt','w') as f:
	    f.write('Validation accuracy\nMean: '+str(np.mean(all_val_acc))+' Variance:'+ str(np.var(all_val_acc)))

