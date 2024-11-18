'''
Name: Sophie Turner.
Date: 14/11/2024.
Contact: st838@cam.ac.uk.
See where the random forest's performance drops with the smallest J rates.
'''

import time
import numpy as np
import file_paths as paths
import constants as con
import prediction_fns_numpy as fns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# File paths.
data_file = f'{paths.npy}/low_res_yr_500k.npy'

print('\n10th tests, 77 to 86\n')
start = time.time()
print('Loading data')
data = np.load(data_file)
print(data.shape)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Split the dataset by Fast-J cutoff pressure.
#print('Removing upper stratosphere')
#data, _ = fns.split_pressure(data)
# Remove zero flux.
#print('Removing night times.')
#data = data[:, np.where(data[10] > 0)].squeeze()

# Input data.
inputs = data[con.phys_no_o3]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[con.J_core]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# Just for simplicity while doing that loop there.
inputs_all = inputs.copy()
targets_all = targets.copy()

# Loop through increments.
tops = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for top in tops:
  bottom = top - 0.1
  print(f'\nTrain and test on {int(bottom * 100)}% to {int(top * 100)}% of J rates\n')
  
  # Increments of range of J rates.
  inputs, targets, _ = fns.only_range(inputs_all, targets_all, targets_all, bottom, top, 1000)
  print('Inputs:', inputs.shape)
  print('Targets:', targets.shape)
  
  # TTS.
  in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1)

  # Make the regression model.
  model = RandomForestRegressor(n_estimators=20, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)

  # Train the model.
  start = time.time()
  print('Training model')
  model.fit(in_train, out_train)

  # Test.
  start = time.time()
  print('Testing model')
  out_pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)

  # View performance.
  fns.show(out_test, out_pred, maxe, mse, mape, smape, r2)

