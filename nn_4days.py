'''
Name: Sophie Turner.
Date: 28/5/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a neural network using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''

import torch
import numpy as np
from torch import nn

# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
train_file = f'{dir_path}/4days.npy'
test_file = f'{dir_path}/20171201.npy'
name_file = f'{dir_path}/idx_names'

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.linspace(0,13,14, dtype=int)
NO2 = 15
HCHOm = 18 # Molecular product.
H2O2 = 73
O3 = 77 # O(1D) product.

# Training data.
train_days = np.load(train_file)

# Testing data.
test_day = np.load(test_file)

# Input features.
features = [1,7,8,9,10]
in_train = torch.tensor(train_days[features])
in_test = torch.tensor(test_day[features])

# Output target.
target_idx = NO2
out_train = torch.tensor(train_days[target_idx])
out_test = torch.tensor(test_day[target_idx])

print('in_train:', in_train)
print('in_test:', in_test)
print('out_train:', out_train)
print('out_test:', out_test)

print('in_train:', in_train.shape)
print('in_test:', in_test.shape)
print('out_train:', out_train.shape)
print('out_test:', out_test.shape)
