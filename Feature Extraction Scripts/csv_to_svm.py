# use CSV feature file to train and save a SVM model
import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
import os
from sklearn.svm import SVC

output_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/'


train_data_cqcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_cqcc.csv'
train_data_mfcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc.csv'
train_label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

dev_data_cqcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_cqcc.csv'
dev_data_mfcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
dev_label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'


train = [train_data_cqcc_path,train_data_mfcc_path,train_label_path,'train']
dev = [dev_data_cqcc_path,dev_data_mfcc_path,dev_label_path,'dev']

features = ['cqcc','mfcc']
	


for idx, feature in enumerate(features):


	feature_len = 650
	if(feature == 'cqcc'):
		feature_len = 3000

	print('Reading Data File')	

	labels = pd.read_csv(train[2], delimiter = " ", header=None)
	labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']
	data = pd.read_csv(train[idx],index_col=0)
	X = data.iloc[:, 0:feature_len].values
	y = data[[str(feature_len)]].values
	svm = SVC(probability=True)	

	print('Training Started')
	
	t0 = time.time()
	svm.fit(X, y)
	print('SVM trained, time spend:', time.time() - t0)
	pickle.dump(svm, open(output_path + 'svm_model__' + feature + '.svm', 'wb'))



