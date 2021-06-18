import scipy.io
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import h5py
import mat73
import pickle



bon_path = '/scratch/fk99/ae1028/genuineFeatureCell_orig.mat'
spoof_path = '/scratch/fk99/ae1028/spoofFeatureCell_orig.mat'
output_path = '/scratch/fk99/ae1028/'

bonafide_data = scipy.io.loadmat(bon_path)
spoof_data = mat73.loadmat(spoof_path)

gmm_bon = GMM(n_components = 144, covariance_type='diag',n_init = 50,verbose=1) 
gmm_sp  = GMM(n_components = 144, covariance_type='diag',n_init = 50,verbose=1) 


bon_data = []
spo_data = []
target_num_rows = 50 #minimum duration of each file in this dataset


for item in bonafide_data['genuineFeatureCell']:
    feature = np.array([np.array(xi) for xi in item[0]])   
    if(target_num_rows - feature.shape[1]) > 0:
        feature = np.concatenate((feature, feature[:,:target_num_rows - feature.shape[1]]), axis=1)
        if(target_num_rows - feature.shape[1]) > 0:
            feature = np.concatenate((feature, np.array([[0.]*(target_num_rows - feature.shape[1])]*(90))),axis=1)
    else:
        feature = feature[:,:target_num_rows]  
    bon_data.append(feature.reshape(-1))

bon_data = np.vstack(bon_data)
print('Training Bonafide Gmm ')
gmm_bon.fit(bon_data)
pickle.dump(gmm_bon, open(output_path + 'cqcc_50_bon.gmm', 'wb'))

for item in spoof_data['spoofFeatureCell']:
    feature = np.array([np.array(xi) for xi in item[0]])   
    if(target_num_rows - feature.shape[1]) > 0:
        feature = np.concatenate((feature, feature[:,:target_num_rows - feature.shape[1]]), axis=1)
        if(target_num_rows - feature.shape[1]) > 0:
            feature = np.concatenate((feature, np.array([[0.]*(target_num_rows - feature.shape[1])]*(90))),axis=1)
    else:
        feature = feature[:,:target_num_rows]  
    spo_data.append(feature.reshape(-1))

spo_data = np.vstack(spo_data)

print('Training Spoof Gmm ')
gmm_sp.fit(spo_data)
pickle.dump(gmm_sp, open(output_path + 'cqcc_50_spoof.gmm', 'wb'))

