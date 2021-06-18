# build train and save CNN model
# model is alexnet

import pandas as pd
import os
import librosa
import librosa.display
from tensorflow import keras
import time
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
mirrored_strategy = tf.distribute.MirroredStrategy()

name_suffix_train = 'amplitutde-LA-train-500n-1024w-256'
name_suffix_dev = 'amplitutde-LA-dev-500n-1024w-256'

X = np.load('/scratch/fk99/ae1028/X-' + name_suffix_train +'.npy')
y = np.load('/scratch/fk99/ae1028/y-'+  name_suffix_train +'.npy')
filenames = np.load('/scratch/fk99/ae1028/filenames-'+  name_suffix_train +'.npy')

X_dev = np.load('/scratch/fk99/ae1028/X-'+ name_suffix_dev +'.npy')
y_dev = np.load('/scratch/fk99/ae1028/y-'+ name_suffix_dev +'.npy')

dev_protocol = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

model_name = name_suffix_train +'-50'


# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 
yy = yy[:,0]

# split the dataset 
#x_train, x_test, y_train, y_test = train_test_split(X, yy[:,0], test_size=0.2, random_state = 42)

num_rows = X[0].shape[0] # 84
num_columns = X[0].shape[1] # 300
num_channels = 1
filter_size = 2

print('Input Shape is:', num_rows,num_columns,num_channels)

x_train = []
for u in X:
    x_train.append(np.reshape(u, (num_rows, num_columns, num_channels)))
x_train = np.asarray(x_train)



x_dev = []
for u in X_dev:
    x_dev.append(np.reshape(u, (num_rows, num_columns, num_channels)))
x_dev = np.asarray(x_dev)

le = LabelEncoder()
y_dev = to_categorical(le.fit_transform(y_dev)) 
y_dev = y_dev[:,0]

activation_func = 'relu'

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation=activation_func, input_shape=(num_rows, num_columns, num_channels)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation=activation_func, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation=activation_func, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation=activation_func, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation=activation_func, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation=activation_func),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation=activation_func),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(lr=0.0001), metrics=['accuracy'])
model.summary()

model.fit(x_train,yy,
          epochs=50,
          use_multiprocessing = True,
          validation_data=(x_dev, y_dev),
          validation_freq=1
        )

model.save(model_name)
print('--------------------------------------')
print('--------------------------------------')
print('-----------MODEL SAVED----------------')
print('--------------------------------------')
print('--------------------------------------')