import os
import numpy as np
import pickle
import argparse
import sys
import time
import pandas as pd

root_path = 'Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/'
train_features_path = '/Users/asimadnan/Desktop/Mres/Experiments/testfiles/la_train_mfcc_cqcc.pkl'
dev_features_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/19-04-2021_23-21dev_2.pkl'
output_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/'

# sample small file to test evrything, comment this line after all tests run correctly
# train_features_path = '/Users/asimadnan/Desktop/Mres/Experiments/testfiles/04-04-2021_15-09_dev_1.pkl'
# dev_features_path = '/Users/asimadnan/Desktop/Mres/Experiments/testfiles/04-04-2021_15-09_dev_1.pkl'
###################


def extract_feature(pkl_path,feature_type):
    max_len = 50  # 1.25 seconds  # check the timesteps of cqcc and mfcc 
    X = []
    y = []
    i=1
    with open(pkl_path, 'rb') as infile:
        data = pickle.load(infile)
        total_files = len(data)
        for feat_cqcc, feat_mfcc, label,filename in data:

            features = []
            feature_block = ''
           
            if feature_type == 'mfcc':
                feature_block = feat_mfcc
            elif feature_type == 'cqcc':
                feature_block = feat_cqcc

            if hasattr(feature_block, "__len__"):
	            if len(feature_block) > max_len:
	                features = feature_block[:max_len]
	            elif len(feature_block) < max_len:
	                features = np.concatenate((feature_block, np.array([[0.]*num_dim]*(max_len-len(feature_block)))), axis=0)
	            
	            if (i%100 == 0):
	                print( ((i/total_files)*100), ' % done' )
	            print(i)
	            i+=1
	            X.append(features.reshape(-1))
	            y.append([label,filename])
        return X,y


features = ['cqcc']
datasets = [dev_features_path]
dataset_name = ['dev']


for feature in features:
	print('-----------------------------------')
	print('Starting on ', feature)
	X,y = extract_feature(dev_features_path,feature)
	csv_features = np.append(np.array(X),np.array(y),1)
	pd.DataFrame(csv_features).to_csv(os.path.join(output_path + 'dev_' + feature + '.csv'))




