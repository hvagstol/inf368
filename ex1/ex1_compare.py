# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:50:00 2019

@author: hvagstol@gmail.com
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import h5py

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json


from helpers import performance_eval, plot_samples, training_eval, save_summary, save_json, load_json, random_init

with open('output/simple_history.json', 'r') as f:
	simple_history = json.loads(f.read())

with open('output/lenet_history.json', 'r') as f:
	lenet_history = json.loads(f.read())

sns.set()
sns.set_style("dark")

# summarize history for training accuracy
plt.clf()
plt.plot(simple_history['acc'])
plt.plot(lenet_history['acc'])
plt.title('simple vs lenet training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['simple', 'lenet'], loc='upper left')
plt.savefig('output/simple_lenet_training_accuracy.png', bbox_inches='tight')

# summarize history for validation accuracy
plt.clf()
plt.plot(simple_history['val_acc'])
plt.plot(lenet_history['val_acc'])
plt.title('simple vs lenet validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['simple', 'lenet'], loc='upper left')
plt.savefig('output/simple_lenet_validation_accuracy.png', bbox_inches='tight')

# summarize history for training loss
plt.clf()
plt.plot(simple_history['loss'])
plt.plot(lenet_history['loss'])
plt.title('simple vs lenet training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['simple', 'lenet'], loc='upper left')
plt.savefig('output/simple_lenet_training_loss.png', bbox_inches='tight')

# summarize history for training loss
plt.clf()
plt.plot(simple_history['val_loss'])
plt.plot(lenet_history['val_loss'])
plt.title('simple vs lenet validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['simple', 'lenet'], loc='upper left')
plt.savefig('output/simple_lenet_validation_loss.png', bbox_inches='tight')

