import numpy as np
import torch

import pandas as pd
import h5py
import time
import os

def LoadOASIS(root='./'):
    sample_x = 100
    subjects = 372
    subjects_for_test = 64
    subjects_for_val = 64
    no_good_comp = 53
    tc = 120

    ntrain_samples = 244
    ntest_samples = 64

    hf_path = os.path.join(root, 'OASIS3_AllData.h5')

    hf = h5py.File(hf_path, 'r')
    data2 = hf.get('OASIS3_dataset')
    data2 = np.array(data2)
    print(data2.shape)
    data2 = data2.reshape((subjects, sample_x, tc))
    data = data2
    print('Reshape we need:', data.shape)

    indices_path = os.path.join(root, 'correct_indices_GSP.csv')
    print(indices_path)
    df = pd.read_csv(indices_path, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]
    print('Reshape we finalized:', finalData.shape)

    FNC = np.zeros((subjects, 1378))
    corrM = np.zeros((subjects, no_good_comp, no_good_comp))
    for i in range(subjects):
        corrM[i, :, :] = np.corrcoef(finalData[i])
        M = corrM[i, :, :]
        FNC[i, :] = M[np.triu_indices(53, k=1)]

    print(FNC.shape)

    index_path = os.path.join(root, 'index_array_labelled_OASIS3.csv')
    df = pd.read_csv(index_path, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    labels_path = os.path.join(root, 'labels_OASIS3.csv')
    df = pd.read_csv(labels_path, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)

    # all_labels = all_labels - 1

    FNC = FNC[index_array, :]
    all_labels = all_labels[index_array.long()]
    finalData=finalData[index_array, :, :]
    corrM = corrM[index_array, :, :]
    return  corrM, finalData, FNC, all_labels


if __name__ == "__main__":
    '''tsData: time series data for 372 subjects and 53 components per subj
    flattened_fc: flattened upper triangle
    labels: class labels'''
   
    corrMatrix, tsData, flattened_fc, labels = LoadOASIS('./OASIS')
    print('..............Done with file reading and preprocessing.............')
    print(f'Correlation Matrix: {corrMatrix.shape}\nData: {tsData.shape}\nFlattened FC: {flattened_fc.shape}\nLabels: {labels.shape}')