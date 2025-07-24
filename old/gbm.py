'''
Name: Sophie Turner
Date: 24/9/24
Contact: st838@cam.ac.uk
Make gradient boosted decision trees to predict J rates from UKCA output data.
'''

import time
import psutil
import numpy as np
import constants as con
import file_paths as paths
import prediction_fns_numpy as fns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingRegressor as gbr

# Memory usage at start of program.
mem_start = psutil.virtual_memory().used / con.GB

print('\nGRADIENT BOOSTING TEST\n')

 # File path.
data_file = f'{paths.npy}/20150115.npy'

start = time.time()
print('Loading data')
data = np.load(data_file)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Split the dataset by Fast-J cutoff pressure.
print('Removing upper stratosphere')
data, _ = fns.split_pressure(data)

# Input data.
inputs = data[con.phys_best]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[con.NO2]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = tts(inputs, targets, test_size=0.05, random_state=con.seed, shuffle=False) 

# Maybe set max_features too? 
params = {'loss': 'absolute_error', 
          'learning_rate': 0.05, 
	  'n_estimators': 500, 
	  'subsample': 0.8, 
	  'max_depth': 10, 
	  'min_samples_leaf': 100, 
	  'verbose': 1, 
	  'max_leaf_nodes': 1000, 
	  'validation_fraction': 0.05, 
	  'n_iter_no_change': 50, 
	  'tol': 1e-6, 
	  'random_state': con.seed}
#model = MultiOutputRegressor(gbr(**params))
model = gbr(**params)

# Train the model.
start = time.time()
print('Training model')
model.fit(in_train, out_train)
end = time.time()
print(f'Training the model took {round(end-start)} seconds.')

tree = model.estimators_[0]
nodes = tree.tree_.node_count
depth = tree.tree_.max_depth
feature = tree.tree_.feature[0]
thresh = tree.tree_.threshold[0]
print(f'Tree 1 has {nodes} nodes and is {depth} branches tall.')
print(f'Tree 1 begins splitting on feature {feature} at a threshold of {thresh}.')

# Test.
start = time.time()
print('Testing model')
out_pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)
print('out_test:', out_test.shape)
print('out_pred:', out_pred.shape)
end = time.time()
print(f'Testing the model took {round(end-start)} seconds.')

# Save.
np.save(f'{paths.npy}/out_pred.npy', out_pred)
np.save(f'{paths.npy}/out_test.npy', out_test)

# Memory usage at end.
mem_end = psutil.virtual_memory().used / con.GB
mem = round(mem_end - mem_start, 3)
print(f'Memory usage: {mem} GB.')

print(f'R2: {r2}')
print(f'MSE: {mse}')
# View performance.
fns.show(out_test, out_pred, maxe, mse, r2)
