import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from helpers import DataGenerator, performance_eval


# load data
taxa = pd.read_csv('../../data/zooscannet/ZooScanSet/taxa.csv', usecols=["objid","taxon"])

# get the distribution
taxa_distribution = taxa['taxon'].value_counts()

# reduce the data to only the 40 most common categories
top_taxa_labels = taxa_distribution[:40]
reduced = taxa.loc[taxa['taxon'].isin(top_taxa_labels.index)]

# downsample the data to a fully balanced set
downsampled = pd.DataFrame()
for label in top_taxa_labels.index:
    
    this_category = reduced.loc[reduced['taxon'] == label].sample(n=1000) #using a low n for demo, running code uses 3455
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

params_test = {'dim': (224,224),
          'batch_size': 64,
          'n_classes': 40,
          'n_channels': 3,
          'shuffle': False}


base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 40 classes
predictions = Dense(40, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

training_generator = DataGenerator(X_train, y_train, le, lb, **params)
validation_generator = DataGenerator(X_test, y_test, le, lb, **params)
testing_generator = DataGenerator(X_test, y_test, le, lb, testing=True, **params_test)

# train the model on the new data for a few epochs
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,epochs=5)

y_fit = model.predict_generator(generator=testing_generator, use_multiprocessing=True, workers=6)

test_length = np.shape(y_fit)[0]

performance_eval('resnet', le.inverse_transform(y_fit.argmax(axis=1)),np.array(y_test.values).ravel())[0:test_length]
print('done')