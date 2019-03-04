import numpy as np
import keras
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
from tensorflow import set_random_seed
from keras.models import model_from_json, load_model

def random_init(number):
    """
    helper to set numpy and tensorflow random seed.
    input - seed number used for both numpy and tensorflow
    """
    set_random_seed(number)

def training_eval(results, title='resnet'):
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


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for image data, based on:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Modified to be more pandas friendly and to import images using pillow
    """

    'Generates data for Keras'
    def __init__(self, list_IDs, labels, lenc, lbin, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=93, shuffle=False, testing=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.lenc = lenc
        self.lbin = lbin
        self.basepath = '../../data/zooscannet/ZooScanSet/imgs/'
        self.testing = testing

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs.index[k] for k in indexes]                
                
        # Generate data
        if (self.testing):
             X=self.__data_generation(list_IDs_temp)
             return X
        else:
            X, y = self.__data_generation(list_IDs_temp)
            return X, y
        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_image(self, infilename ) :
        img = Image.open( infilename )
        img.load()
        #not good resizing, prbly
        img = img.resize((224,224), Image.BICUBIC)        
        data = np.asarray( img, dtype="int32").reshape(224,224,1)
        return data
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=str)

        # Generate data
        #print(self.labels)
        #print(list_IDs_temp)
        for i, ID in enumerate(list_IDs_temp):
            # Store sample            
            path = self.basepath + str(self.labels.loc[ID]['taxon']) + '/' + str(self.list_IDs.loc[ID]['objid']) + '.jpg'
            print(path)
            img = self.load_image(path)
            X[i,] = img

            if (self.testing == False):
                # Store class, first need to one hot encode it using the received encoders
                named_class=self.labels.loc[ID]['taxon']
                labeled_class=self.lenc.transform([named_class])
                onehot_class =  self.lbin.transform([labeled_class])                        
                y[i] = onehot_class
            
        if (self.testing):
            return X
        else:
            return X, y