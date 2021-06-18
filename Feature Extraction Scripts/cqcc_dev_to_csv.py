# script to convert matlab generated cqcc files to csv files so its easier to use it with models

import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
from sklearn.mixture import GaussianMixture as GMM
import os
import scipy.io
import h5py
import mat73

feature = 'cqcc'
env = 'gadi'
env = 'local'
target_num_rows = 50

if feature == 'mfcc':
	 bon_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc_bon.gmm'
	# spf_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc_spoof.gmm'
	# data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
else:
    if env == 'gadi':
        dev_filename = '/scratch/fk99/ae1028/CQCC_filelist_dev.mat'
        data_path = '/scratch/fk99/ae1028/devFeatureCell_train.mat'
        dev_protocol = '/scratch/fk99/ae1028/ASVspoof2019.LA.cm.dev.trl.txt'
        output_path = '/scratch/fk99/ae1028/'
    else:
        dev_filename = '/Users/asimadnan/Desktop/Mres/ASVspoof_2019_baseline_CM_v1/AllTrainData/CQCC_filelist_dev.mat'
        data_path = '/Users/asimadnan/Desktop/Mres/ASVspoof_2019_baseline_CM_v1/AllTrainData/CQCC_devFeatureCell_train.mat'        
        dev_protocol = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        output_path = '/Users/asimadnan/Desktop/Mres/Experiments/'


dev_feature_data = mat73.loadmat(data_path)
dev_file_names = scipy.io.loadmat(dev_filename)
labels = pd.read_csv(dev_protocol, delimiter = " ", header=None)
labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']

dev_data = []

for idx,item in enumerate(dev_feature_data['devFeatureCell'][1:100]):
    feature = np.array([np.array(xi) for xi in item])   
    if(target_num_rows - feature.shape[1]) > 0:
        feature = np.concatenate((feature, feature[:,:target_num_rows - feature.shape[1]]), axis=1)
    else:
        feature = feature[:,:target_num_rows]
    
    
    feature = feature.reshape(-1)
    feature = np.append(feature,dev_file_names['filelist'][idx][0][0])
    dev_data.append(feature)


feature_len = len(dev_data[0]) -1

del dev_feature_data

dev_data = pd.DataFrame(dev_data)
dev_data = dev_data.rename({(feature_len): 'AUDIO_FILE_NAME'}, axis=1)
dev_data = pd.merge(dev_data, labels, on='AUDIO_FILE_NAME')

dev_data.to_csv('dev_data_50_cqcc.csv')
