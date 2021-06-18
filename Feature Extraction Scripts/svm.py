import numpy as np
import pandas as pd
import argparse
import csv
import time
import pickle
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# mfcc_v1.csv, mfcc_test.csv
#python3 svm.py --train_data_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/mfcc_v1.csv' --train_label_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt' --eval_data_labels '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt' --eval_data_path '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/mfcc_v1.csv' --output_path '/Users/asimadnan/Desktop/Mres/Experiments/Models/'

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
	svm = pickle.load(open(model_path,'rb'))
else:
	#Train Model
	train_data = pd.read_csv(train_data_path, delimiter = ",", header=None)
	train_labels = pd.read_csv(train_label_path, delimiter = " ", header=None)
	train_labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']
	X = train_data.drop([0,601], axis=1)
	y = train_data[[601]]
	svm = SVC(probability=True)
	print('Training Started')
	t0 = time.time()
	svm.fit(X, y)
	print('SVM trained, time spend:', time.time() - t0)
	current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 
	pickle.dump(svm, open(output_path + current_time +'svm_model' + '.svm', 'wb'))
	
	


# Test with eval files
eval_data = pd.read_csv(eval_data_path, delimiter = ",", header=None)
eval_labels =  pd.read_csv(eval_labels_path, delimiter = " ", header=None)

X_eval = eval_data.drop([0,601], axis=1)
y_eval = eval_data[[601]]

y_score = svm.decision_function(X_eval)

# real target values
y = eval_data[[601]].values.tolist()



cm_score = cm_score[['AUDIO_FILE_NAME','ATTACK_ID','KEY','SCORE']]

current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 
filename = output_path + current_time + 'cm_score.txt'

cm_score.to_csv(filename,index=False,sep=" ",header=False)
cm_score

print('cm_score file saved at:',filename)














