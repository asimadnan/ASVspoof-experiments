# script to get scores from saved CNN model.

import pandas as pd
from tensorflow import keras
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



train_file_name = 'amplitutde-LA-train-500n-1024w-256'
name_suffix_dev = 'amplitutde-LA-dev-500n-1024w-256'

#load CNN model
model = tf.keras.models.load_model(train_file_name + '-50')


#load feature files
X = np.load('/scratch/fk99/ae1028/X-'+ name_suffix_dev +'.npy')
y = np.load('/scratch/fk99/ae1028/y-'+ name_suffix_dev +'.npy')
filenames = np.load('/scratch/fk99/ae1028/filenames-'+ name_suffix_dev +'.npy')

#load dev protocal file
dev_protocol = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

# not really needed here, added just for logging purposes
num_rows = X[0].shape[0] # 84
num_columns = X[0].shape[1] # 300
num_channels = 1
filter_size = 2


# reshaping input into a ndarray
x_dev = []
for u in X:
    x_dev.append(np.reshape(u, (num_rows, num_columns, num_channels)))
x_dev = np.asarray(x_dev)

# encoding labels, needed to create score files for dev set
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# get predictions
results = model.predict(x_dev,verbose=0)

# creating list of filenames
filenames_temp = []
for f in filenames:
    filenames_temp.append(f.split('.')[0])
filenames = np.asarray(filenames_temp)

# compiling scores into a df
# print(results)
scores = pd.DataFrame({'SCORE': results[:,0], 'AUDIO_FILE_NAME': filenames})

#using label file to create file for tdcf function.
labels = pd.read_csv(dev_protocol, delimiter = " ", header=None)
labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']
pd.merge(scores, labels, on='AUDIO_FILE_NAME')[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']].to_csv(train_file_name + '_scores.csv',index=False,sep=" ",header=False)


print("Written results of this model to:" + train_file_name + '_scores.csv')





