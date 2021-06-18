import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
import os
from sklearn.svm import SVC

feature = 'mfcc'

if feature == 'mfcc':
	svm = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/svm_model__mfcc.svm'
	data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
else:
	svm = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/svm_model__cqcc.svm'
	data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_cqcc.csv'

#dev labels because testing models on dev data

label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
output_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/'



svm_model = pickle.load(open(svm,'rb'))

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

y_score = svm_model.decision_function(raw_data)
labels[['SCORE']] = y_score

labels[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']].to_csv(os.path.join(output_path + feature + 'svm_dev_scores.csv'),index=False,sep=" ",header=False)


