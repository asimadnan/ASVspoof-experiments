import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,Conv1D
from keras.models import model_from_json

def extract_data(data_path,label_path,feature_type):
    data_set = pd.read_csv(data_path, delimiter = ",",index_col=0)
    labels = pd.read_csv(label_path, delimiter = " ", header=None)
    labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']

    feature_len = 650

    if (feature_type == 'cqcc'):
    	feature_len = 3000

    
    data_set = data_set.rename({str(feature_len + 1): 'AUDIO_FILE_NAME'}, axis=1)
    data_set = pd.merge(data_set, labels, on='AUDIO_FILE_NAME')
    
    return data_set.iloc[:, 0:feature_len].values,data_set.iloc[:, feature_len+1:data_set.shape[1]]

feature = 'cqcc'

if feature == 'mfcc':
	feature_len = 650
	svm = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/svm_model__mfcc.svm'
	test_data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
	train_data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc.csv'
else:
	feature_len = 3000
	svm = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/svm_model__cqcc.svm'
	test_data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_cqcc.csv'
	train_data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_cqcc.csv'


dev_label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
train_label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
output_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/'



raw_data,labels = extract_data(train_data_path,train_label_path,feature)
x_train, y_train = raw_data,labels[['KEY']].values

raw_data,labels = extract_data(test_data_path,dev_label_path,feature)
x_test, y_test = raw_data,labels[['KEY']].values


a = (y_train == 'bonafide')
flat_list = [item for sublist in a for item in sublist]
y_train = flat_list
a = (y_test == 'bonafide')
flat_list = [item for sublist in a for item in sublist]
y_test = flat_list
y_train = 1*np.array(y_train)
y_test = 1*np.array(y_test)

batch_size = 4
epochs = 1
num_classes = 2
print('Starting Building Model')



model = Sequential()
model.add(Dense(64, input_dim=feature_len, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(34, activation='relu'))
model.add(Dense(1, activation='softplus'))

model.compile(loss=keras.losses.categorical_crossentropy,  metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))


model.save(os.path.join(output_path + feature +'cnn_model'),save_format='h5')




