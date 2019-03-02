import numpy as np
import keras
from PIL import Image
      
class DataGenerator(keras.utils.Sequence):
    """
    Data generator for image data, based on:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Modified to be more pandas friendly and to import images using pillow
    """

    'Generates data for Keras'
    def __init__(self, list_IDs, labels, lenc, lbin, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=93, shuffle=False):
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
        self.basepath = '../../data/ZooScanSet/imgs/'

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs.index[k] for k in indexes]                
        
        # we need to pass the index somehow as well..
        
        # Generate data
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
            #print(path)
            img = self.load_image(path)
            X[i,] = img

            # Store class, first need to one hot encode it using the received encoders
            named_class=self.labels.loc[ID]['taxon']
            labeled_class=self.lenc.transform([named_class])
            onehot_class =  self.lbin.transform([labeled_class])                        
            y[i] = onehot_class
            
        #print(y)
        return X, y