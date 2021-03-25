from python_speech_features import mfcc
from CQCC.cqcc import cqcc
import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# 
# python3 -W ignore preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/TEST_FILES --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/new_dataa.pkl

# train
# python3 preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/flac --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/train_new.pkl
# 
# eval
# python3 preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/flac --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/eval_new.pkl
# 
# 
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--label_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.pkl')
# parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()

def extract_cqcc(x, fs):
    # INPUT SIGNAL
    x = x.reshape(x.shape[0], 1)  # for one-channel signal 
    # print(x.shape)
    # fs: 16000
    # x: (64244,)
    # PARAMETERS
    B = 96
    fmax = fs/2
    fmin = fmax/2**9
    d = 16
    cf = 19
    ZsdD = 'ZsdD'
    # COMPUTE CQCC FEATURES
    CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec, absCQT = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)
    return CQcc, fmax, fmin


# read in labels
filename2label = {}
for line in open(args.label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label

total_files = len(os.listdir(args.data_path))
i = 1

feats = []
t_loop_start = time.time()
for filepath in os.listdir(args.data_path):

    filename = filepath.split('.')[0]
    if filename not in filename2label: # we skip speaker enrollment stage
        continue
    
    t_row_start = time.time()

    label = filename2label[filename]
    #print("filename:", os.path.join(args.data_path, filepath))
    sig, rate = sf.read(os.path.join(args.data_path, filepath))
    #print("rate:", rate)
    feat_cqcc, fmax, fmin = extract_cqcc(sig, rate)
    #print("feat cqcc:", feat_cqcc.shape)
    numframes = feat_cqcc.shape[0]
    winstep = 0.005
    winlen =  (len(sig) - winstep*rate*(numframes-1))/rate
    feat_mfcc = mfcc(sig,rate,winlen=winlen,winstep=winstep, lowfreq=fmin,highfreq=fmax)      # number of frames * number of cep
    # if args.feature_type == "cqcc":
    #     feat = extract_cqcc(sig, rate)
    # elif args.feature_type == "mfcc":
    #     feat = mfcc(sig, rate)    
    print("feat mfcc:", feat_mfcc.shape)
    feats.append((feat_cqcc, feat_mfcc, label,filename))
    t_row_end = time.time()

    row_time = t_row_end - t_row_start
    files_left = (total_files- i)
    total_time = t_row_end - t_loop_start
    row_avg = total_time/i
    est_time_left = row_avg * (total_files - i)

    print(i, 'of', total_files, 'Processed,',round( ((i/total_files)*100),2 ),'%', 
              ',Est Time Left:', round(est_time_left/60,2),
              'Mins, time spent so far',round(((t_row_end - t_loop_start)/60),2),
              'Avg Per Row Time',round(row_avg,2), 's')
    i=i+1

t_loop_end = time.time()
#print("number of instances:", len(feats))

with open(args.output_path, 'wb') as outfile:
    pickle.dump(feats, outfile)