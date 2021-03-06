import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
import numpy as np
from tensorflow import set_random_seed
from keras.models import model_from_json, load_model

def random_init(number):
    """
    helper to set numpy and tensorflow random seed.
    input - seed number used for both numpy and tensorflow
    """
    set_random_seed(number)

def plot_samples(data, title='samples', width=3,height=3):

	"""
    plots and saves samples
    input - data: samples to plot, size = w*h
    		title: applied to plot and filename
    		width, height: subplot matrix size

    """

	subplot_no = 1
	for sample in data:
		plt.subplots_adjust(hspace=0.8, wspace=0.8)
		plt.subplot(width,height,subplot_no)
		subplot_no = subplot_no +1
		plt.suptitle(title)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(sample, cmap='Greys')    
	plt.savefig('output/' + title + '.png', bbox_inches='tight')


def training_eval(results, title='classifier'):
    """
    generate plots of training and validation loss for network training
    
    input - results: handle from model
    		title: name of model
    """
  
    # summarize history for accuracy
    plt.clf()
    plt.plot(results.history['acc'])
    plt.plot(results.history['val_acc'])
    plt.title(title + ' training accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig('output/'+title+'_training_accuracy.png', bbox_inches='tight')
    plt.savefig('output/'+title+'_training_accuracy.pdf', bbox_inches='tight')
    
    # summarize history for loss
    plt.clf()
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title(title+' training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig('output/'+title+'_training_loss.png', bbox_inches='tight')
    plt.savefig('output/'+title+'_training_loss.pdf', bbox_inches='tight')

    mean_val_loss = '%.4f' % np.mean(results.history['val_loss'])
    print('Mean validation loss after training ' + mean_val_loss)
    with open('output/'+title+'val_loss.txt','w') as f:
        f.write(mean_val_loss)



def performance_eval(title, y_fit, y_target):
    """
    reports on performance of a given data set vs a given target
    input - title, string to be used for labeling plots and text files
            y_fit, output from model
            y_target, target classes of the data
    """

    report = classification_report(y_target, y_fit)
    with open('output/' +title + '_classification.txt', 'w') as file:
        file.write(report)
    print(report)
    fig, ax = plt.subplots(figsize=(10,10))
    mat = confusion_matrix(y_target, y_fit)
    acc = accuracy_score(y_target,y_fit)
    accuracy_report = str('\n\n{} accuracy: {:.5f}'.format(title, acc))
    with open('output/' +title + '_classification.txt', 'a') as file:
        file.write(accuracy_report)
    print(accuracy_report)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='BuGn')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig('output/' + title + '_confusion.png', bbox_inches='tight')
    plt.savefig('output/' +title + '_confusion.pdf' ,bbox_inches='tight')


def save_summary(model, title):
	"""
	saves model summary to disk and displays to to terminal
	input: title to be used in filename
	"""

	with open('models/'+title+'_summary.txt', 'w') as f:
		model.summary(print_fn=lambda x: f.write(x + '\n'))
	print(model.summary())


def save_json(model, title):
	"""
	saves model architecture to disk in json format
	input: title to be used in filename
	"""
	with open('models/'+title+'_architecture.json', 'w') as f:
		f.write(model.to_json())

def load_json(title):
    """
    loads model from models/title_architecture.json
    input: title to be used in filename
    return: model from json file
    """

    with open('models/'+title+'_architecture.json', 'r') as f:
        json = f.read()
    model = model_from_json(json)
    return model