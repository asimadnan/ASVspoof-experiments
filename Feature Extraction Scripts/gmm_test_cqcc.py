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

if feature == 'mfcc':
	 bon_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc_bon.gmm'
	# spf_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc_spoof.gmm'
	# data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
else:
	bon_model_path = '/scratch/fk99/ae1028/models/cqcc__bon.gmm'
	spf_model_path = '/scratch/fk99/ae1028/models/cqcc__spoof.gmm'
	dev_filename = '/scratch/fk99/ae1028/CQCC_filelist_dev.mat'
	data_path = '/scratch/fk99/ae1028/devFeatureCell_train.mat'

#dev labels because testing models on dev data

dev_protocol = '/scratch/fk99/ae1028/ASVspoof2019.LA.cm.dev.trl.txt'
output_path = '/scratch/fk99/ae1028/'



gmm_bon = pickle.load(open(bon_model_path,'rb'))
gmm_sp  = pickle.load(open(spf_model_path,'rb'))


# def extract_data(data_path,label_path,feature_type):
#     data_set = pd.read_csv(data_path, delimiter = ",",index_col=0)
#     labels = pd.read_csv(label_path, delimiter = " ", header=None)
#     labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']

#     feature_len = 650

#     if (feature_type == 'cqcc'):
#     	feature_len = 3000

    
#     data_set = data_set.rename({str(feature_len + 1): 'AUDIO_FILE_NAME'}, axis=1)
#     data_set = pd.merge(data_set, labels, on='AUDIO_FILE_NAME')
    
#     return data_set.iloc[:, 0:feature_len].values,data_set.iloc[:, feature_len+1:data_set.shape[1]]




# raw_data,labels = extract_data(data_path,label_path,feature)


dev_feature_data = mat73.loadmat(data_path)
dev_file_names = scipy.io.loadmat(dev_filename)
labels = pd.read_csv(dev_protocol, delimiter = " ", header=None)

labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']
target_num_rows = 160


dev_data = []
for idx,item in enumerate(dev_feature_data['devFeatureCell']):
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


 
X = dev_data.iloc[:, 0:feature_len].values
y = dev_data['KEY']


predic_scores = []
counter = len(X)

for i in range(counter):
    if (i % 50 == 0):
        print('Evaluating Bon sample at',i/counter * 100, '%')
    x_obs = [X[i]]
    bscore = gmm_bon.score(x_obs)
    sscore = gmm_sp.score(x_obs)

    #predb.append(np.exp(bscore)-np.exp(sscore))
    predic_scores.append(bscore-sscore)

preds = np.asarray(predic_scores)

labels[['SCORE']] = predic_scores

labels[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']].to_csv(os.path.join(output_path + 'gmm_dev_scores.csv'),index=False,sep=" ",header=False)


