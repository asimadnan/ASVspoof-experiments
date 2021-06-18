import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
from sklearn.mixture import GaussianMixture as GMM
import os

feature = 'cqcc'

if feature == 'mfcc':
	bon_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc_bon.gmm'
	spf_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_mfcc_spoof.gmm'
	data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_mfcc.csv'
else:
	bon_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_cqcc_bon.gmm'
	spf_model_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/train_cqcc_spoof.gmm'
	data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/dev_cqcc.csv'

#dev labels because testing models on dev data

label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
output_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/'



gmm_bon = pickle.load(open(bon_model_path,'rb'))
gmm_sp  = pickle.load(open(spf_model_path,'rb'))


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

predic_scores = []
counter = len(raw_data)

for i in range(counter):
    if (i % 50 == 0):
        print('Evaluating Bon sample at',i/counter * 100, '%')
    X = [raw_data[i]]
    bscore = gmm_bon.score(X)
    sscore = gmm_sp.score(X)

    #predb.append(np.exp(bscore)-np.exp(sscore))
    predic_scores.append(bscore-sscore)

preds = np.asarray(predic_scores)

labels[['SCORE']] = predic_scores

labels[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']].to_csv(os.path.join(output_path + feature + '_dev_scores.csv'),index=False,sep=" ",header=False)


