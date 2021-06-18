# extract pickle file to generate  dataframe
import numpy as np
import pickle
import pandas as pd
import time

def extract_feature(pkl_path,feature_type):
    max_len = 50  # 1.25 seconds  # check the timesteps of cqcc and mfcc 
    X = []
    y = []
    i=1

    with open(pkl_path, 'rb') as infile:
        data = pickle.load(infile)
        #print(len(data))
        for feat_cqcc, feat_mfcc, label,filename in data:
            features = []
            feature_block = ''

           
            if feature_type == 'mfcc':
                feature_block = feat_mfcc
            elif feature_type == 'cqcc':
                feature_block = feat_cqcc

            if feature_block is not 0:
                if len(feature_block) > max_len:
                    features = feature_block[:max_len]
                elif len(feature_block) < max_len:
                    features = np.concatenate((feature_block, np.array([[0.]*num_dim]*(max_len-len(feature_block)))), axis=0)
                
                if i%100 == 0:
                    print((i/len(data)*100) ,"% Done")
                i+=1
                X.append(features.reshape(-1))
                y.append([label,filename])
        return X,y



def main():
    start = time.time()
    pkl_path = '/Users/asimadnan/Desktop/Mres/Experiments/testfiles/04-04-2021_15-09_dev_1.pkl'
    output_path = '/Users/asimadnan/Desktop/Mres/Experiments/testfiles/'
    data_name = 'la_dev_'
    feature_type = 'cqcc'

    X,y = extract_feature(pkl_path,feature_type)
    data_cqcc = np.append(np.array(X),np.array(y),1)
    current_time = time.strftime("%d-%m-%Y_%H-%M", time.localtime()) 
    pd.DataFrame(data_cqcc).to_csv(output_path + current_time + data_name + feature_type + '.csv', header='None')
    print('File Saved: ' + output_path + current_time + data_name + feature_type + '.csv')
    print(f'elapsed time: {time.time() - start} Seconds')


if __name__ == '__main__':
    main()


