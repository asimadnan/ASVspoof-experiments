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


feature = 'cqcc'

if feature == 'mfcc':
	model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/mfcccnn_model_3'
	data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
else:
	model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/cqcccnn_model'
	data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_cqcc.csv'



label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
output_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/'

model = keras.models.load_model(model_path)


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

raw_data,labels = extract_data(data_path,label_path,feature)
x_test, y_test = raw_data,labels[['KEY']].values
a = (y_test == 'bonafide')
flat_list = [item for sublist in a for item in sublist]
y_test = flat_list
y_test = 1*np.array(y_test)

test_prob = model.predict(x_test)

labels[['SCORE']] = test_prob
labels[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']].to_csv(os.path.join(output_path + feature + 'nnet_dev_scores.csv'),index=False,sep=" ",header=False)



