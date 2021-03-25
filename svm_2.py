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
parser.add_argument("--feature_type", required=True, type=str, help='mfcc or cqcc')
args = parser.parse_args()



model_path = args.model_path
train_data_path = args.train_data_path
train_label_path = args.train_label_path
eval_data_path = args.eval_data_path
eval_labels_path = args.eval_data_labels
output_path = args.output_path
feature_type = args.feature_type


if model_path:
	svm = pickle.load(open(model_path,'rb'))
else:	
	max_len = 50  # 1.25 seconds  # check the timesteps of cqcc and mfcc 
	lens = []
	X = []
	y = []
	file_names = []


	with open(train_data_path, 'rb') as infile:
	    data = pickle.load(infile)
	    for feat_cqcc, feat_mfcc, label,filename in data:

	        if feature_type == "cqcc":
	            feats = feat_cqcc
	        elif feature_type == "mfcc":
	            feats = feat_mfcc
	            
	        num_dim = feats.shape[1]
	        if len(feats) > max_len:
	            feats = feats[:max_len]
	        elif len(feats) < max_len:
	            # padded with zeros
	            feats = np.concatenate((feats, np.array([[0.]*num_dim]*(max_len-len(feats)))), axis=0)
	        X.append(feats.reshape(-1))
	        y.append(label)
	        file_names.append(filename)


	svm = SVC(probability=True)

	print('Training Started')
	t0 = time.time()

	svm.fit(X, y)

	print('SVM trained, time spend:', time.time() - t0)
	current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 
	pickle.dump(svm, open(output_path + current_time +'svm_model' + '.svm', 'wb'))


#Test with eval files...
