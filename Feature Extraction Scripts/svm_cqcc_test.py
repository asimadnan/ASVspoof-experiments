import numpy as np
import pandas as pd
import time
import pickle
from sklearn.svm import SVC
import scipy.io
import h5py
import mat73
import os


dev_feature = '/Users/asimadnan/Desktop/Mres/ASVspoof_2019_baseline_CM_v1/AllTrainData/CQCC_devFeatureCell_train.mat'
dev_filename = '/Users/asimadnan/Desktop/Mres/ASVspoof_2019_baseline_CM_v1/AllTrainData/CQCC_filelist_dev.mat'
dev_protocol = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
model_path = '/Users/asimadnan/Desktop/Mres/Experiments/04-05-2021_04-51svm_model.svm'
output_path = '/Users/asimadnan/Desktop/Mres/Experiments/'

# dev_feature = '/scratch/fk99/ae1028/devFeatureCell_train.mat'
# dev_filename = '/scratch/fk99/ae1028/CQCC_filelist_dev.mat'
# dev_protocol = '/scratch/fk99/ae1028/ASVspoof2019.LA.cm.dev.trl.txt'
# model_path = '/scratch/fk99/ae1028/models/04-05-2021_04-51svm_model.svm'
#output_path = '/scratch/fk99/ae1028/'



dev_feature_data = mat73.loadmat(dev_feature)
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



#dev_data = extract_feature_dev(dev_feature_data['devFeatureCell'][1:200],160,dev_file_names['filelist'][1:200])

feature_len = len(dev_data[0]) -1

del dev_feature_data

dev_data = pd.DataFrame(dev_data)
dev_data = dev_data.rename({(feature_len): 'AUDIO_FILE_NAME'}, axis=1)
dev_data = pd.merge(dev_data, labels, on='AUDIO_FILE_NAME')


 
X = dev_data.iloc[:, 0:feature_len].values
y = dev_data['KEY']

del dev_data


svm_model = pickle.load(open(model_path,'rb')) 
y_score = svm_model.decision_function(X)
labels[['SCORE']] = y_score


labels[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']].to_csv(os.path.join(output_path + 'cqcc_svm_dev_scores.csv'),index=False,sep=" ",header=False)
