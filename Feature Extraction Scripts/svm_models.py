import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
from sklearn.svm import SVC
import scipy.io
import h5py
import mat73



bon_path = '/Users/asimadnan/Desktop/Mres/ASVspoof_2019_baseline_CM_v1/AllTrainData/genuineFeatureCell_train.mat'
spoof_path = '/Users/asimadnan/Desktop/Mres/ASVspoof_2019_baseline_CM_v1/AllTrainData/spoofFeatureCell_train.mat'

bonafide_data = scipy.io.loadmat(bon_path)
spoof_data = mat73.loadmat(spoof_path)

bon_data = []
spo_data = []
target_num_rows = 160 #minimum duration of each file in this dataset




for item in bonafide_data['genuineFeatureCell'][1:100]:
    feature = np.array([np.array(xi) for xi in item[0]])   
    if(target_num_rows - feature.shape[1]) > 0:
        feature = np.concatenate((feature, feature[:,:target_num_rows - feature.shape[1]]), axis=1)
        if(target_num_rows - feature.shape[1]) > 0:
            feature = np.concatenate((feature, np.array([[0.]*(target_num_rows - feature.shape[1])]*(90))),axis=1)
    else:
        feature = feature[:,:target_num_rows]  
    bon_data.append(feature.reshape(-1))

bon_data = np.c_[ bon_data, np.array(['bonafide']*len(bon_data)) ]



for item in spoof_data['spoofFeatureCell'][1:100]:
    feature = np.array([np.array(xi) for xi in item[0]])   
    if(target_num_rows - feature.shape[1]) > 0:
        feature = np.concatenate((feature, feature[:,:target_num_rows - feature.shape[1]]), axis=1)
        if(target_num_rows - feature.shape[1]) > 0:
            feature = np.concatenate((feature, np.array([[0.]*(target_num_rows - feature.shape[1])]*(90))),axis=1)
    else:
        feature = feature[:,:target_num_rows]  
    spo_data.append(feature.reshape(-1))

spo_data = np.c_[ spo_data, np.array(['spoof']*len(spo_data)) ]

joined_data = np.r_[ bon_data, spo_data ]
class_label = joined_data[:, -1] # for last column
dataset = joined_data[:, :-1] # for all but last column


svm = SVC(probability=True)

print('Training Started')
t0 = time.time()

svm.fit(dataset, class_label)


print('SVM trained, time spend:', time.time() - t0)
current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 
output_path = '/scratch/fk99/ae1028/'
pickle.dump(svm, open(output_path + current_time +'svm_model' + '.svm', 'wb'))