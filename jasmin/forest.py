'''
Name: Sophie Turner
Date: 24/9/24
Contact: st838@cam.ac.uk
Make a random forest to predict J rates from UKCA output data.
'''

import time
import psutil
import numpy as np
import file_paths as paths
import prediction_fns as fns
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import train_test_split as tts


print('RANDOM FOREST TEST: NO2 & O3 benchmark\n')

# Constants.
GB = 1000000000
NO2 = 16
O3 = 78
J_core = [16,18,19,20,24,28,30,31,32,33,51,52,66,70,71,72,73,74,75,78,79,80,81,82]
phys_all = np.arange(15, dtype=np.int16)
phys_no_o3 = np.arange(14, dtype=np.int16)
phys_best = [1, 7, 8, 9, 10, 14]
seed = 6

# Memory usage at start of program.
mem_start = psutil.virtual_memory().used / GB

 # File path.
data_file = f'{paths.data}/low_res_yr_2015.npy'

start = time.time()
print('Loading data')
data = np.load(data_file)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Remove night.
#data = data[:, np.where(data[10] > 0)].squeeze()
# Remove upper portion where Fast-J might not work.
#data = data[:, np.where(data[7] > 20)].squeeze()  

# Input data.
inputs = data[phys_no_o3]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[[16, 18]]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = tts(inputs, targets, test_size=0.05, random_state=seed, shuffle=False) 

params = {'n_estimators': 20,
          'max_features': 0.3,
	  'max_leaf_nodes': 10000, 
          'n_jobs': 20,
	  'random_state': seed,
	  'verbose': 2,
	  'max_samples': 0.2}
'''
# 1 Tree.
params = {'n_estimators': 1,
	  'random_state': seed,
	  'verbose': 1}	  
'''
model = rf(**params)

# Train the model.
start = time.time()
print('Training model')
model.fit(in_train, out_train)
end = time.time()
print(f'Training the model took {round(end-start)} seconds.')

# Get info about the trees in our forest for explainability.
print(model)
for i in range(len(model.estimators_)):
  tree = model.estimators_[i]
  nodes = tree.tree_.node_count
  depth = tree.tree_.max_depth
  print(f'Tree {i+1} has {nodes} nodes and is {depth} branches tall.')

# Test.
start = time.time()
print('Testing model')
out_pred, mse, mape, r2 = fns.test(model, in_test, out_test)
print('out_test:', out_test.shape)
print('out_pred:', out_pred.shape)
end = time.time()
print(f'Testing the model took {round(end-start)} seconds.')

# Save.
np.save(f'{paths.results}/out_pred_bothbenc.npy', out_pred)
np.save(f'{paths.results}/out_test_bothbenc.npy', out_test)

# Memory usage at end.
mem_end = psutil.virtual_memory().used / GB
mem = round(mem_end - mem_start, 3)
print(f'Memory usage: {mem} GB.')

print(f'R2: {r2}')
print(f'Max E: {maxe}')
print(f'MAPE: {mape}')
