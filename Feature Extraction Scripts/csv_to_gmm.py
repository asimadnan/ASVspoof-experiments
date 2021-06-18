# use CSV feature file to traina dn save a GMM model

import os
import numpy as np
import pickle
import sys
import time
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
#import samplerate

output_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/'


train_data_cqcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_cqcc.csv'
train_data_mfcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc.csv'
train_label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

dev_data_cqcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_cqcc.csv'
dev_data_mfcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
dev_label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'



# train_data_mfcc_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/test__mfcc.csv'
# train_data_cqcc_path= '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/test__cqcc.csv'


train = [train_data_cqcc_path,train_data_mfcc_path,train_label_path,'train']
dev = [dev_data_cqcc_path,dev_data_mfcc_path,dev_label_path,'dev']

features = ['cqcc','mfcc']
datasets = [ dev ]


for dataset in datasets:
	for idx, feature in enumerate(features):
		

		if(dataset[3] == 'train' and feature == 'cqcc'):
			continue

		print('Features',feature, 'Dataset', dataset[3])
		feature_len = 650
		if(feature == 'cqcc'):
			feature_len = 3000
			
		labels = pd.read_csv(dataset[2], delimiter = " ", header=None)
		labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']

		print(dataset[idx])

		data = pd.read_csv(dataset[idx],index_col=0)
		data = data.rename({str(feature_len + 1): 'AUDIO_FILE_NAME'}, axis=1)
		data = pd.merge(data, labels, on='AUDIO_FILE_NAME')
		
		X = data.iloc[:, 0:feature_len+1]
		
		bondata = X.loc[X[str(feature_len)] == 'bonafide'].iloc[:, 0:feature_len]
		spoofdata = X.loc[X[str(feature_len)] == 'spoof'].iloc[:, 0:feature_len]


		gmm_bon = GMM(n_components = 144, covariance_type='diag',n_init = 50,verbose=1) # min shape[0] = 135 # max = 1112
		gmm_sp  = GMM(n_components = 144, covariance_type='diag',n_init = 50,verbose=1)  # min shape[0] = 64  # max = 1318

		t0 = time.time()
		print('Training Bonafide Gmm on',feature, dataset[3])

		gmm_bon.fit(bondata)
		print('Bonafide gmm trained, time spend:', time.time() - t0)
		pickle.dump(gmm_bon, open(output_path + dataset[3] + '_' + feature + '__bon.gmm', 'wb'))


		t0 = time.time()
		print('Training Spoof Gmm on',feature, dataset[3])
		gmm_sp.fit(spoofdata)
		print('Spoof gmm trained, time spend:', time.time() - t0)
		pickle.dump(gmm_sp, open(output_path + dataset[3] + '_' + feature + '__spoof.gmm', 'wb'))


