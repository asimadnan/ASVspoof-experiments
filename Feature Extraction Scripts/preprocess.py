import librosa
import numpy as np
import pandas as pd
import os
import csv
import argparse
import time


# python3 preprocess.py --data_path ./LA/ASVspoof2019_LA_train/flac 
# --output_path ./data/train_mfcc.csv 
# --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
#
#

# python3 preprocess.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/flac/ --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/mfcc.csv --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt

# python3 preprocess.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/flac/ --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/mfcc.csv --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt

# python3 preprocess.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/flac/ --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/mfcc.csv --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--label_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.csv')
args = parser.parse_args()


#train_files = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/sample_train/'
#train_labels = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

train_files = args.data_path
train_labels = args.label_path
file_out_path = args.output_path


la_train_labels = pd.read_csv(train_labels, delimiter = " ", header=None)
la_train_labels.columns = ['SPEAKER_ID','AUDIO_FILE_NAME','ENVIRONMENT_ID','ATTACK_ID','KEY']

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms

max_len = 30
features = []
max_len = 30

total_files = len(os.listdir(train_files))
i = 1
with open(file_out_path, "w") as f:
    writer = csv.writer(f,delimiter=',',lineterminator='\n')

    t_loop_start = time.time()
    for filepath in os.listdir(train_files):
        t_row_start = time.time()
        filename = filepath.split('.')[0]
        if filename not in la_train_labels.values[:, 1]: # checking if filename exists in label file
            print('File not Found in Label File')
            continue
        file_index = np.where(la_train_labels.values== filename)[0][0]    
        #print("Filename:", os.path.join(train_files, filepath))
        signal, sampling_rate = librosa.load(train_files + filepath)
        #print(sampling_rate)
        mfcc = librosa.feature.mfcc(y=signal, sr=sampling_rate)

        feature_t = np.transpose(mfcc)
        num_dim = feature_t.shape[1]
        length = len(feature_t)
        if length > max_len:
            feature_f = feature_t[:max_len]
        elif length < max_len:
            # add zero's if len < max lenght, taken this max_len from an existing implemntation, maybe need to change
            feature_f = np.concatenate((feature_t, np.array([[0.]*num_dim]*(max_len-length))), axis=0)
        feature_row = feature_f.reshape(-1).tolist()
        #print(len(feature_row))
        feature_row.append(la_train_labels.values[file_index][4])
        feature_row.insert(0, filename)

        features.append(feature_row)
        writer.writerow(feature_row)
        
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
    print('Total Time Taken ', round((t_loop_end-t_loop_start)/60,2), 'Seconds, Avg time per row',round(((t_loop_end-t_loop_start)/i),2))

        #file_out_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/'

# with open(file_out_path, "w") as f:
#     writer = csv.writer(f,delimiter=',',lineterminator='\n')
#     writer.writerows(features)
    