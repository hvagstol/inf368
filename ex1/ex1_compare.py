# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 18:00:00 2019

@author: hvagstol@gmail.com
"""
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

