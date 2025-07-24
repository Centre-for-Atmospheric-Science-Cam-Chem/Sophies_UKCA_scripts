'''
Name: Sophie Turner.
Date: 17/7/2024.
Contact: st838@cam.ac.uk
Experiment with NN designs for UKCA data.
'''

import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score, mean_absolute_percentage_error


GB = 1000000000
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at start of program: {mem} GB.')


# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
name_file = f'{dir_path}/idx_names'
train_file1 = f'{dir_path}/20170601.npy'
train_file2 = f'{dir_path}/20110115.npy' 

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
NO2 = 16
HCHO = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.

print('Loading numpy data.')

# Pick a smaller random sample of the data.
data = np.load(train_file1)
size = len(data[0])
indices = np.arange(size, dtype=np.int32)
indices = np.random.choice(indices, 10000)
indices = np.sort(indices)
data = data[:, indices]
print('data shape', data.shape)

# Select input and output data.
# Chose the 5 best inputs from linear selection for now. Do proper NN feature selection later.
features = [1,7,8,9,10]
target_idx = NO2
inputs = data[features]
targets = data[target_idx]

# Train-test-split.
in_train, in_test, out_train, out_test = tts(inputs, targets, test_size=0.1, train_size=0.9)

print('\nin_train:', in_train.shape)
print('in_test:', in_test.shape)
print('out_train:', out_train.shape)
print('out_test:', out_test.shape)

exit()
