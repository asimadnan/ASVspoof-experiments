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
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count



# python3 multi-preproc.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/flac --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/ --dataset dev_2



# 
# python3 -W ignore preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/TEST_FILES --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/ --dataset test

# train
# python3 preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/flac --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/ --dataset train
# 
# eval
# python3 preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/flac --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_eval/ --dataset eval

# dev
# python3 preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/flac --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/ --dataset dev

# samll subset
# python3 preprocess_2.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/sample_train --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/ --dataset train
# 


parser = argparse.ArgumentParser()
parser.add_argument("--label_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--data_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.pkl')
parser.add_argument("--output_file_name", required=True, type=str, help='dev eval or train')

# python3 multi-preproc.py --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/flac --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --output_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/output_data/ --output_file_name dev_2

# parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()

# dataset = args.dataset

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

def process_one_file(filepath):

    filename = filepath.split('.')[0]
    if filename not in filename2label: # we skip speaker enrollment stage
        return (0,0,0,filename)
    label = filename2label[filename]
    sig, rate = sf.read(os.path.join(data_path, filepath))
    feat_cqcc, fmax, fmin = extract_cqcc(sig, rate)
    numframes = feat_cqcc.shape[0]
    winstep = 0.005
    winlen =  (len(sig) - winstep*rate*(numframes-1))/rate
    feat_mfcc = mfcc(sig,rate,winlen=winlen,winstep=winstep, lowfreq=fmin,highfreq=fmax)      # number of frames * number of cep
    return (feat_cqcc, feat_mfcc, label,filename)


# label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
# data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/sample_dev'
# output_path = '/Users/asimadnan/Desktop/Mres/Experiments/testfiles/'


#python3 multi-preproc.py --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/sample_dev  --output_path /Users/asimadnan/Desktop/Mres/Experiments/testfiles/ --output_file_name test_data

# --label_path data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --data_path data/LA/ASVspoof2019_LA_dev/sample_dev --output_path data/LA/output_files --output_file_name test_data

# --label_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --data_path /Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_dev/sample_dev  --output_path /Users/asimadnan/Desktop/Mres/Experiments/testfiles/ --output_file_name test_data

#python multi-preproc.py --label_path data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --data_path data/LA/ASVspoof2019_LA_dev/sample_dev --output_path data/LA/output_files --output_file_name test_data

label_path = args.label_path
data_path = args.data_path
output_path = args.output_path
output_file_name = args.output_file_name



def chunks(lst, n):
#"""Yield successive n-sized chunks from lst."""
    list = []
    for i in range(0, len(lst), n):
        list.append(lst[i:i + n])
    return list



def extract_features(x):
    #t_start = time.time() 
    feats = []
    for filename in x:
        feats.append(process_one_file(filename))
        print('Est time left:',((avg_time*total_files) - start_time-timer() )/60 , 'minutes'  )
        #total_count.append(1)
        #print(len(total_count))
    #print('Time Per Chunk:',(time.time() - t_start), 'Seconds')
    return feats

flatten = lambda t: [item for data in t for item in data]


filename2label = {}
for line in open(label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label


avg_time = (13109/25380)
total_files = len(os.listdir(data_path))
start_time = timer()

def main():
    # read in labels

    avg_time = (13109/25380)




    total_files = len(os.listdir(data_path))
    #num_of_chunks = round(total_files/4)
    # with a test of chunk size from 2 to n/2, where n is total files,
    # it was found that the smalles chunk size was the fastest overall time.
    # tested on 100 files, may differ on 24k+ files.
    file_lists = chunks(os.listdir(data_path),2)

    start = timer()
    print(f'starting computations on {cpu_count()} cores')
    print( 'Total Chunks:',len(file_lists))
    print( 'Chunks Size:',len(file_lists[0]))

    with Pool() as pool:
        res = pool.map(extract_features, file_lists)
        #print(res)



    current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 
    with open(output_path + current_time + output_file_name + '.pkl', 'wb') as outfile:
        pickle.dump(flatten(res), outfile)

    end = timer()
    print(f'elapsed time: {end - start}')

#LA DEV
#elapsed time: 12973.868796231


#LA Train
# elapsed time: 13109.089001766
# total files: 25380

if __name__ == '__main__':
    main()