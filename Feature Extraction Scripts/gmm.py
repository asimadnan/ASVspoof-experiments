
import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d



# python3 gmm.py --model_path '/Users/asimadnan/Desktop/Mres/Experiments/Models/' --train_data_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/mfcc_v1.csv' --train_label_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt' --eval_data_labels '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt' --eval_data_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/mfcc_test.csv' 

# python3 gmm.py --train_data_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/mfcc_v1.csv' --train_label_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt' --eval_data_labels '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt' --eval_data_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/mfcc_v1.csv' --output_path '/Users/asimadnan/Desktop/Mres/Experiments/Models/'


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=False, type=str, help='path to existing GMM models, expecting model names bon_train.gmm & sp_train.gmm')
parser.add_argument("--train_data_path", required=True, type=str, help='path to train dataset file create by preprocessing.py')
parser.add_argument("--train_label_path", required=True, type=str, help='path to train label file in asv data')
parser.add_argument("--eval_data_labels", required=True, type=str, help='path to eval labels file create in asv data')
parser.add_argument("--eval_data_path", required=True, type=str, help='path to eval dataset file create by preprocessing.py')
parser.add_argument("--output_path", required=True, type=str, help='path to output model and results, ./data/train.csv')
args = parser.parse_args()



model_path = args.model_path
train_data_path = args.train_data_path
train_label_path = args.train_label_path
eval_data_path = args.eval_data_path
eval_labels_path = args.eval_data_labels
output_path = args.output_path

if model_path:
	gmm_bon = pickle.load(open(model_path + 'bon_train' + '.gmm','rb'))
	gmm_sp  = pickle.load(open(model_path + 'sp_train' + '.gmm','rb'))
else:
	#Building Models
	train_data = pd.read_csv(train_data_path, delimiter = ",", header=None)
	train_labels = pd.read_csv(train_label_path, delimiter = " ", header=None)
	train_labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']
	X = train_data.drop([0,601], axis=1)
	y = train_data[[601]]
	gmm_bon = GMM(n_components = 144, covariance_type='diag',n_init = 50) # min shape[0] = 135 # max = 1112
	gmm_sp  = GMM(n_components = 144, covariance_type='diag',n_init = 50)  # min shape[0] = 64  # max = 1318

	bondata = []
	spdata = []
	feature_type = 'mfcc'

	bondata = train_data.loc[train_data[601] == 'bonafide']
	spdata = train_data.loc[train_data[601] == 'spoof']

	bondata = bondata.drop([0,601], axis=1).sample(150)
	spdata = spdata.drop([0,601], axis=1).sample(150)

	Xbon = np.vstack(bondata.values)
	Xsp = np.vstack(spdata.values)

	current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 

	t0 = time.time()
	gmm_bon.fit(Xbon)
	print('Bon gmm trained, time spend:', time.time() - t0)
	pickle.dump(gmm_bon, open(output_path + current_time +'_bon_train' + '.gmm', 'wb'))


	t0 = time.time()
	gmm_sp.fit(Xsp)
	print('Sp gmm trained, time spend:', time.time() - t0)
	pickle.dump(gmm_sp, open(output_path + current_time + '_sp_train' + '.gmm', 'wb'))


	print('GMM model created')




	# // create a models
	# load traning data and labels
	# Train Models
	


# //load eval_file	

eval_data = pd.read_csv(eval_data_path, delimiter = ",", header=None)
eval_labels =  pd.read_csv(eval_labels_path, delimiter = " ", header=None)

# convert data to be ready for testing by models.


# seperating bonafide and spoof observations
# in real scneario we won't be able to do this with eval set, becuase we don't know what the label is
bondata_eval = eval_data.loc[eval_data[601] == 'bonafide']
spdata_eval = eval_data.loc[eval_data[601] == 'spoof']


bondata_eval = bondata_eval.drop([0,601], axis=1).values
spdata_eval = spdata_eval.drop([0,601], axis=1).values


predb = []
preds = []
j_bon = len(bondata_eval)
k_sp  = len(spdata_eval)


for i in range(j_bon):
    if (i % 50 == 0):
        print('Evaluating Bon sample at',i/j_bon * 100, '%')
    X = [bondata_eval[i]]
    bscore = gmm_bon.score(X)
    sscore = gmm_sp.score(X)

    #predb.append(np.exp(bscore)-np.exp(sscore))
    predb.append(bscore-sscore)

for i in range(k_sp):
    if (i % 50 == 0):
        print('Evaluating Sp sample at',i/k_sp * 100, '%')
    X = [spdata_eval[i]]
    bscore = gmm_bon.score(X)
    sscore = gmm_sp.score(X)

    #preds.append(np.exp(bscore)-np.exp(sscore))
    preds.append(bscore-sscore)

predb1 = np.asarray(predb)
preds1 = np.asarray(preds)


# getting file name for all files
bondata_eval = eval_data.loc[eval_data[601] == 'bonafide'][[0]]
spdata_eval = eval_data.loc[eval_data[601] == 'spoof'][[0]]


bondata_eval["score"] = predb1
spdata_eval["score"] = preds1

# combining all obs
final_score = pd.concat([spdata_eval,bondata_eval])

# getting predicted score in a list
y_score = final_score[['score']].values.tolist()

# real target values
y = eval_data[[601]].values.tolist()

# Calc EER

# fpr, tpr, thresholds = roc_curve(y, y_score, pos_label='spoof')
# eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
# print ('EER', eer)

final_score.columns = ['AUDIO_FILE_NAME','SCORE']
final_score.loc[final_score['SCORE'] > 0, 'Prediction'] = 'bonafide'
final_score.loc[final_score['SCORE'] < 0, 'Prediction'] = 'spoof'

eval_labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']
cm_score = pd.merge(final_score, eval_labels, on="AUDIO_FILE_NAME")	

cm_score = cm_score[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']]

current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 
filename = output_path + current_time + 'cm_score.txt'

cm_score.to_csv(filename,index=False,sep=" ",header=False)
cm_score

print('cm_score file saved at:',filename)








	