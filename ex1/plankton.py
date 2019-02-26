import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from PIL import Image
import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

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
        y = np.empty((self.batch_size, 93), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #print (i, ID)
            #print(self.labels[i])
            X[i,] = self.load_image('../../data/ZooScanSet/imgs/'+ str(ID[1]) + '/' + str(ID[0]) + '.jpg')

            # Store class
            y[i] = self.labels[i]

        return X, y


taxa = np.array(pd.read_csv('../../data/ZooScanSet/taxa.csv', usecols=["objid","taxon"]))

# Create a one hot encoder and set it up with the categories from the data
ohe = OneHotEncoder(dtype='int8',sparse=False)
taxa_labels = np.unique(taxa[:,1])
ohe.fit(taxa_labels.reshape(-1,1))

# Create a categorical list of targets for each sample
y = ohe.transform(taxa[:,1].reshape(-1,1))

# Get string label from one hot encoder, might be useful later
labels= ohe.inverse_transform(y)


# split into training and test data, ensuring stratification

X_use, X_dontuse, y_use, y_dontuse = train_test_split(taxa,y, test_size=0.999, random_state=42, stratify=taxa[:,1])
X_train, X_test, y_train, y_test = train_test_split(X_use,y_use, test_size=0.1, random_state=42, stratify=y_use[:,1])


# Parameters
params = {'dim': (224,224),
          'batch_size': 64,
          'n_classes': 93,
          'n_channels': 3,
          'shuffle': False}

base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(93, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

training_generator = DataGenerator(X_train, y_train, **params)
validation_generator = DataGenerator(X_test, y_test, **params)

# train the model on the new data for a few epochs
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=2,
                    epochs=2)