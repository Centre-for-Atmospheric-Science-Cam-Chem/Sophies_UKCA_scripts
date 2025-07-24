'''
Name: Sophie Turner
Date: 24/9/24
Contact: st838@cam.ac.uk
Make a linear support vector machine to predict J rates from UKCA output data.
'''

import time
import psutil
import numpy as np
import file_paths as paths
from sklearn.svm import LinearSVR
import prediction_fns as fns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts


print('Linear SVM test\n')

# Constants.
GB = 1000000000
NO2 = 16
HCHOr = 18
J_core = [16,18,19,20,24,28,30,31,32,33,51,52,66,70,71,72,73,74,75,78,79,80,81,82]
phys_all = np.arange(0, 15, dtype=np.int16)
phys_best = [1, 7, 8, 9, 10, 14]
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

# Smaller subsample.
data = fns.sample(data, 2000000)

# Input data.
inputs = data[phys_all]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[HCHOr]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# Standardisation.
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs, targets)

# TTS.
in_train, in_test, out_train, out_test = tts(inputs, targets, test_size=0.1, random_state=seed, shuffle=False) 

# Configure the model.
model = LinearSVR(random_state=seed)

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
np.save(f'{paths.results}/out_pred_SVM.npy', out_pred)
np.save(f'{paths.results}/out_test_SVM.npy', out_test)

# Memory usage at end.
mem_end = psutil.virtual_memory().used / GB
mem = round(mem_end - mem_start, 3)
print(f'Memory usage: {mem} GB.')

print(f'R2: {r2}')
print(f'MSE: {mse}\n')
