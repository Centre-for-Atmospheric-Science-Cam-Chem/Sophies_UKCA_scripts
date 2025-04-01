'''
Name: Sophie Turner.
Date: 7/2/2025.
Contact: st838@cam.ac.uk.
Check the data and calcs for scaleds.
'''
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def check_data(inputs, targets, preds=None):
  # Inputs and targets should be the same.
  print('\nInputs:', inputs.shape)
  print('Range:', np.min(inputs), np.max(inputs))
  print('Sample 10:', inputs[10, 10])
  print('\nTargets:', targets.shape)
  print('Range:', np.min(targets), np.max(targets))
  print('Sample 10:', targets[10, 10])
  if preds is not None:
    # Preds can be slightly different.
    print('\nPreds:', preds.shape)
    print('Sample 10:', targets[10, 10])
    print('Range:', np.min(preds), np.max(preds))
    print('R2:', r2_score(targets, preds))
    

# Cross reference training dataset with random forest trained data and scaled data. 
data_path = f'{paths.npy}/low_res_yr_500k.npy'
models = ['rf', 'rf_scaled']

# Load training dataset.
data = np.load(data_path)

# Input data.
inputs = data[con.phys_main]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)

# Target data.
targets = data[con.J_core]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 

del(data)

# Replicate the same TTS.
_, inputs, _, targets = train_test_split(inputs, targets, test_size=0.1, shuffle=False, random_state=con.seed)

print('\nFresh split from training dataset')
check_data(inputs, targets)

for model in models:
  # Load trained random forest data.
  inputs, targets, preds = fns.load_model_data(model)
  print(f'\n{model}')
  check_data(inputs, targets, preds)
