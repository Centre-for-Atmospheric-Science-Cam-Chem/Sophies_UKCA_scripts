'''
Name: Sophie Turner.
Date: 20/9/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a decision tree using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''

import time
import joblib
import numpy as np
import file_paths as paths
import prediction_fns_numpy as fns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# File paths.
data_file = f'{paths.npy}/20150115.npy'
name_file = f'{paths.npy}/idx_names'

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=np.int16)
J_all = np.arange(15, 85, dtype=np.int16)
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
NO3 = 66
HOCl = 71
H2O2 = 74
O3 = 78 # O(1D) product.
# J rates which are not summed or duplicate fg. with usually zero rates removed.
J_core = [16,18,19,20,24,28,30,31,32,33,51,52,66,70,71,72,73,74,75,78,79,80,81,82]

print('\nDECISION TREE COST COMPLEXITY PRUNING TEST\n')
start = time.time()
print('Loading data')
data = np.load(data_file)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Split the dataset by Fast-J cutoff pressure.
print('Removing upper stratosphere')
data, _ = fns.split_pressure(data)

# Input data.
inputs = data[[1, 7, 8, 9, 10, 14]]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[J_core]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1, random_state=6, shuffle=False) 

# Make the regression model.
model = DecisionTreeRegressor(max_depth=20, max_features=0.3, random_state=6, max_leaf_nodes=100000)

# Train the model.
start = time.time()
print('Growing the tree')
model.fit(in_train, out_train)
print(model)
end = time.time()
print(f'Growing (training) the tree took {round(end-start)} seconds.')

# Get info about pruning complexity constant.
start = time.time()
print('Getting the pruning path.')
path = model.cost_complexity_pruning_path(in_train, out_train)
ccp_alphas = path.ccp_alphas
print('Cost complexity pruning alphas:', ccp_alphas)
end = time.time()
print(f'Getting the pruning path took {round(end-start)} seconds.')

# Test.
start = time.time()
print('Testing model')
out_pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)
end = time.time()
print(f'Testing the model took {round(end-start)} seconds.')

print(r2, maxe, mse)
