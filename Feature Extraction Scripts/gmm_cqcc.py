import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
from sklearn.mixture import GaussianMixture as GMM


train_data_path = '/Users/asimadnan/Desktop/Mres/Experiments/testfiles/04-04-2021_15-57_la_train_cqcc.csv'
train_label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'


train_labels = pd.read_csv(train_label_path, delimiter = " ", header=None)
train_labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']

train_data = pd.read_csv(train_data_path, delimiter = ",",index_col=0)
train_data = train_data.rename({'3001': 'AUDIO_FILE_NAME'}, axis=1)

merged_data = pd.merge(train_data, train_labels, on='AUDIO_FILE_NAME')

X = merged_data.iloc[:, 0:3001]

bondata = X.loc[X['3000'] == 'bonafide']
spoofdata = X.loc[X['3000'] == 'spoof']

bondata = bondata.iloc[:, 0:3000]
spoofdata = spoofdata.iloc[:, 0:3000]

gmm_bon = GMM(n_components = 144, covariance_type='diag',n_init = 50,verbose=1) # min shape[0] = 135 # max = 1112
gmm_sp  = GMM(n_components = 144, covariance_type='diag',n_init = 50,verbose=1)  # min shape[0] = 64  # max = 1318

t0 = time.time()
print('im here')

gmm_bon.fit(bondata)
print('Bon gmm trained, time spend:', time.time() - t0)
pickle.dump(gmm_bon, open(output_path + current_time +'_bon_train' + '.gmm', 'wb'))


t0 = time.time()
gmm_sp.fit(spoofdata,verbose=1)
print('Sp gmm trained, time spend:', time.time() - t0)


pickle.dump(gmm_bon, open('/Users/asimadnan/Desktop/Mres/Experiments/testfiles/cqcc_bon.gmm', 'wb'))
pickle.dump(gmm_sp, open('/Users/asimadnan/Desktop/Mres/Experiments/testfiles/cqcc_spoof.gmm', 'wb'))


