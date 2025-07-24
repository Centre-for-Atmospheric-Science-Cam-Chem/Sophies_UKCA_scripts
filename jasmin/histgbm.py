'''
Name: Sophie Turner
Date: 24/9/24
Contact: st838@cam.ac.uk
Make gradient boosted decision trees to predict J rates from UKCA output data.
'''

import time
import psutil
import numpy as np
import file_paths as paths
import prediction_fns as fns
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import HistGradientBoostingRegressor as hgbr


print('HISTOGRAM GRADIENT BOOSTING TEST: LEAST SQUARES LOSS\n')

# Constants.
GB = 1000000000
NO2 = 16
HCHOr = 18
J_core = [16,18,19,20,24,28,30,31,32,33,51,52,66,70,71,72,73,74,75,78,79,80,81,82]
phys_best = phys_best = [1, 7, 8, 9, 10, 14]
seed = 6

# Memory usage at start of program.
mem_start = psutil.virtual_memory().used / GB

 # File path.
data_file = f'{paths.data}/4days.npy'

start = time.time()
print('Loading data')
data = np.load(data_file)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Split the dataset by Fast-J cutoff pressure.
print('Removing upper stratosphere')
data, _ = fns.split_pressure(data)

# Input data.
inputs = data[phys_best]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[NO2]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = tts(inputs, targets, test_size=0.05, random_state=seed, shuffle=False) 

params = {'loss': 'squared_error',
          'max_iter': 100,
	  'max_leaf_nodes': 2000,
          'learning_rate': 0.05,
	  'l2_regularization': 0.1,
	  'verbose': 1,  
	  'early_stopping': True,
	  'validation_fraction': 0.05,
	  'random_state': seed}

model = hgbr(**params)

# Train the model.
start = time.time()
print('Training model')
model.fit(in_train, out_train)
print(model)
end = time.time()
print(f'Training the model took {round(end-start)} seconds.')

# Test.
start = time.time()
print('Testing model')
out_pred, mse, mape, r2 = fns.test(model, in_test, out_test)
print('out_test:', out_test.shape)
print('out_pred:', out_pred.shape)
end = time.time()
print(f'Testing the model took {round(end-start)} seconds.')

# Save.
np.save(f'{paths.results}/out_pred_HGBMsq.npy', out_pred)
np.save(f'{paths.results}/out_test_HGBMsq.npy', out_test)

# Memory usage at end.
mem_end = psutil.virtual_memory().used / GB
mem = round(mem_end - mem_start, 3)
print(f'Memory usage: {mem} GB.')

print(f'R2: {r2}')
print(f'MSE: {mse}')
