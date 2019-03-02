import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from helpers import DataGenerator, performance_eval


# load data
taxa = pd.read_csv('../../data/ZooScanSet/taxa.csv', usecols=["objid","taxon"])

# get the distribution
taxa_distribution = taxa['taxon'].value_counts()

# reduce the data to only the 40 most common categories
top_taxa_labels = taxa_distribution[:40]
reduced = taxa.loc[taxa['taxon'].isin(top_taxa_labels.index)]

# downsample the data to a fully balanced set
downsampled = pd.DataFrame()
for label in top_taxa_labels.index:
    
    this_category = reduced.loc[reduced['taxon'] == label].sample(n=10) #using a low n for demo, running code uses 3455
    downsampled = pd.concat([downsampled, this_category])
   

X = downsampled[['objid']]
y = downsampled[['taxon']]

# use scikit-learn to prepare encoders to convert labels from
# string to onehot. we will later pass these to the data generator
# as we need the strings for folder referencing as well

le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()

le.fit(np.unique(downsampled['taxon']))
lb.fit(le.transform(np.unique(downsampled['taxon'])))

# split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.6, random_state=42, stratify=y)

# make sure the memory is not clogged with previous data
K.clear_session()

print('Training on '+str(np.shape(X_train)[0]) +' samples')
# Parameters
params = {'dim': (224,224),
          'batch_size': 64,
          'n_classes': 40,
          'n_channels': 3,
          'shuffle': False}

base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(40, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

training_generator = DataGenerator(X_train, y_train, le, lb, **params)
validation_generator = DataGenerator(X_test, y_test, le, lb, **params)
testing_generator = DataGenerator(X_test, y_test, le, lb, testing=True, **params)



# train the model on the new data for a few epochs
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,epochs=1)

y_fit = model.predict_generator(generator=testing_generator, batch_size=64, use_multiprocessing=True, workers=6)

y_test = le.transform(y_test)
y_test = lb.transform(y_test)
performance_eval('resnet',y_fit.argmax(axis=1), y_test.argmax(axis=1))
print('done')