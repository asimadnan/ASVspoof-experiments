# code to extract features from all files in parallel, 
# utilizies multiple cpus to do it faster.
# a liner code is also available.

import os
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
import librosa
import librosa.display
import numpy as np
import pandas as pd
import sys

#Run Instruction
# first command line argument is data type being user, 2nd argument is type of feature being extracted.
# python3 alexnet-multip.py |train or dev| |spectrogram or cqt|

data_dir = '/Users/asimadnan/Desktop/Mres/'
target_num_rows = 500
target_num_rows_2 = 1000
fft = 1024
win_len = 512

data_type = 'train'
if(len(sys.argv) > 1):
    data_type = sys.argv[1]

feature = 'amplitutde'

if(len(sys.argv) > 2):
    feature = sys.argv[2]



out_file_name = feature + '-LA-' + data_type +'-' + str(target_num_rows) + 'n-' + str(fft) + 'w-' +str(win_len)
output_path = ''

# print('Processing', data_type)
# print('Output file name',out_file_name)



if (data_type == 'dev'):
    data_path = data_dir + 'LA/ASVspoof2019_LA_dev/flac'
    labels_path = data_dir + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

if(data_type == 'train'):
    data_path = data_dir + 'LA/ASVspoof2019_LA_train/flac'
    labels_path = data_dir + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

if(data_type == 'eval'):
    data_path = data_dir + 'LA/ASVspoof2019_LA_eval/flac'
    labels_path = data_dir + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

if(data_type == 'eval_2021'):
    data_path = data_dir + 'ASVspoof2021_LA_eval/flac_sample'
    labels_path = data_dir + 'ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt'


filename2label = {}

if data_type == 'eval_2021':
    for file in os.listdir(data_path):
        filename2label[file] = ''
else: 
    for line in open(labels_path):
        line = line.split()
        filename, label = line[1], line[-1]
        filename2label[filename] = label
features = []

total_file = len(os.listdir(data_path))


def get_mel_spectrogram(filename):
    #do processing on a single file based on it name
    try:
        y, sr = librosa.load(data_path + '/' + filename)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
    except Exception as e:
        print("Error encountered while parsing file: ", filename)
        return None 
     
    return S,sr

def get_cqt(filename):
    #do processing on a single file based on it name
    try:
        y, sr = librosa.load(data_path + '/' + filename)
        cqt = np.abs(librosa.cqt(y, sr=sr))
    except Exception as e:
        print("Error encountered while parsing file: ", filename)
        return None 
     
    return cqt,sr

def get_phase(filename):
     #do processing on a single file based on it name
    try:
        y, sr = librosa.load(data_path + '/' + filename)
        raw_fft = librosa.stft(y,n_fft=fft,win_length=win_len)
        phase = np.angle(raw_fft) 
        if(target_num_rows > phase.shape[1]):
                # repeat sample
            while(target_num_rows != phase.shape[1]):
                phase = np.concatenate((phase, phase[:,:target_num_rows - phase.shape[1]]), axis=1)
        else:
            #cut  sample
            phase = phase[:,:target_num_rows]         
    except Exception as e:
        print("Error encountered while parsing file: ", filename)
        return None 
    # print(phase.shape)
    return phase,sr 

def get_amplitutde(filename):
     #do processing on a single file based on it name
    try:
        y, sr = librosa.load(data_path + '/' + filename)
        raw_fft = librosa.stft(y,n_fft=fft,win_length=win_len)
        absolout_fft = np.abs(raw_fft)
        if(target_num_rows > absolout_fft.shape[1]):
                # repeat sample
            while(target_num_rows != absolout_fft.shape[1]):
                absolout_fft = np.concatenate((absolout_fft, absolout_fft[:,:target_num_rows - absolout_fft.shape[1]]), axis=1)
        else:
            #cut  sample
            absolout_fft = absolout_fft[:,:target_num_rows] 


    except Exception as e:
        print("Error encountered while parsing file: ", filename)
        return None 

    # print(absolout_fft.shape) 
    return absolout_fft,sr 
    

def chunks(lst, n):
#"""Yield successive n-sized chunks from lst."""
    list = []
    for i in range(0, len(lst), n):
        list.append(lst[i:i + n])
    return list


counter = 1

# process a list of files
def extract_features(x):
    #t_start = time.time() 
    
   
    feats = []
    for filename in x:
        
        if data_type != 'eval_2021' and filename.split('.')[0] in filename2label:
            class_label = filename2label[filename.split('.')[0]]
            
        if(feature == 'spectrogram'):
            data,sr = get_mel_spectrogram(filename)
        if(feature == 'cqt'):
            data,sr = get_cqt(filename)
        if(feature == 'phase'):
            data,sr = get_phase(filename)
        if(feature == 'phaseAmpl'):
            phase,sr = get_phase(filename)
            amp,sr = get_amplitutde(filename)
            data = np.concatenate((amp,phase),axis=1)
            # target_num_rows = 1000
        if(feature == 'amplitutde'):
                data,sr = get_amplitutde(filename)
        
        
        if(target_num_rows_2 > data.shape[1]):
            # repeat sample
            while(target_num_rows_2 != data.shape[1]):
                data = np.concatenate((data, data[:,:target_num_rows_2 - data.shape[1]]), axis=1)
        else:
            #cut  sample
            data = data[:,:target_num_rows_2] 
        
        print(data.shape,feature)
        if data_type == 'eval_2021':
            feats.append([data,filename])        
        else: 
            feats.append([data, class_label,filename])        
    return feats

def extract_n_features(x):
    #t_start = time.time() 
   
    feats = []
    for filename in x:
        if filename.split('.')[0] in filename2label:
            class_label = filename2label[filename.split('.')[0]]
            
            data_list = []
            spec,sr = get_mel_spectrogram(filename)
            cqt,sr = extract_cqt(filename)

            data_list.append(spec)
            data_list.append(cqt)

            data_clean = []
            for data in data_list:
                if(target_num_rows > data.shape[1]):
                    # repeat sample
                    while(target_num_rows != data.shape[1]):
                        data = np.concatenate((data, data[:,:target_num_rows - data.shape[1]]), axis=1)
                else:
                    #cut  sample
                    data = data[:,:target_num_rows] 
                data_clean.append(data)
            
            col_size = min(data_list[0].shape[0],data_list[1].shape[0])
            feats.append([data_clean[0][:col_size,:],data_clean[1][:col_size,:], class_label,filename])
        
    return feats


flatten = lambda t: [item for data in t for item in data]




def main():
    
    # with a test of chunk size from 2 to n/2, where n is total files,
    # create n chunks of files
    # location to all input files
    #data_path = ''
    file_lists = chunks(os.listdir(data_path),2)
    print('Total Files:',len(file_lists))

    start = timer()
    print(f'starting computations on {cpu_count()} cores')
    print( 'Total Chunks:',len(file_lists))
    print( 'Chunks Size:',len(file_lists[0]))

    with Pool() as pool:
        res = pool.map(extract_features, file_lists)
        # res = pool.map(extract_n_features, file_lists)
        

    print("Feature Extraction Done:")
    if (data_type == 'eval_2021'):
        featuresdf = pd.DataFrame(flatten(res), columns=['feature','filename'])
        X = np.array(featuresdf.feature.tolist())
        filenames = np.array(featuresdf.filename.tolist())
        np.save('X-' + out_file_name, X)
        np.save('filenames-' + out_file_name, filenames)
        print("file saved as: X-",out_file_name)
    else:
        featuresdf = pd.DataFrame(flatten(res), columns=['feature','class_label','filename'])
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())
        filenames = np.array(featuresdf.filename.tolist())

        np.save('X-' + out_file_name, X)
        np.save('y-' + out_file_name, y)
        np.save('filenames-' + out_file_name, filenames)
        print("file saved as: X-",out_file_name)

if __name__ == '__main__':
    main()