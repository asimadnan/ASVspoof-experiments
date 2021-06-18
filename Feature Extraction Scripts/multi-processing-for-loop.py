import os
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
import librosa
import librosa.display
import numpy as np
import pandas as pd




def process_one_file(filename):
    #do processing on a single file based on it name
    try:
        y, sr = librosa.load(data_path + '/' + filename)
        cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
                n_bins=60 * 2, bins_per_octave=12 * 2))
    except Exception as e:
        print("Error encountered while parsing file: ", filename)
        return None 
     
    return cqt,sr

    # return filepath

label_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
data_path = '/Users/asimadnan/Desktop/Mres/ASVSPOOF_DATA/LA/ASVspoof2019_LA_train/sample'
output_path = ''



def chunks(lst, n):
#"""Yield successive n-sized chunks from lst."""
    list = []
    for i in range(0, len(lst), n):
        list.append(lst[i:i + n])
    return list

filename2label = {}
for line in open(label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label
features = []
target_num_rows = 150
total_file = len(os.listdir(data_path))
counter = 1

# process a list of files
def extract_features(x):
    #t_start = time.time() 
   
    feats = []
    for filename in x:
        if filename.split('.')[0] in filename2label:
            class_label = filename2label[filename.split('.')[0]]
            data,sr = process_one_file(filename)
            if(target_num_rows > data.shape[1]):
                # repeat sample
                while(target_num_rows != data.shape[1]):
                    data = np.concatenate((data, data[:,:target_num_rows - data.shape[1]]), axis=1)
            else:
                #cut  sample
                data = data[:,:target_num_rows] 
            
            feats.append([data, class_label,filename])        
    return feats


flatten = lambda t: [item for data in t for item in data]




def main():
    
    # with a test of chunk size from 2 to n/2, where n is total files,
    # create n chunks of files
    # location to all input files
    #data_path = ''
    file_lists = chunks(os.listdir(data_path),2)

    start = timer()
    print(f'starting computations on {cpu_count()} cores')
    print( 'Total Chunks:',len(file_lists))
    print( 'Chunks Size:',len(file_lists[0]))

    with Pool() as pool:
        res = pool.map(extract_features, file_lists)


    featuresdf = pd.DataFrame(flatten(res), columns=['feature','class_label','filename'])
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    filenames = np.array(featuresdf.filename.tolist())

    np.save('X-cqt-sample', X)
    np.save('y-cqt-sample', y)
    np.save('filenames-cqt-sample', filenames)
    # with open(output_path + '.pkl', 'wb') as outfile:
    #     pickle.dump(flatten(res), outfile)


if __name__ == '__main__':
    main()