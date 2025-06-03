'''
Name: Sophie Turner.
Date: 13/5/2025.
Contact: st838@cam.ac.uk.
Compare performace of random forests with and without corrected altitude and SZA.
Not for use with my standard .npy data before May 2025.
'''

import time
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

print('\nOriginal dataset without corrections.')
data_file = f'{paths.npy}/test_day.npy'

#print('\nDataset with the correct altitudes and scaled SZA.')
#data_file = f'{paths.npy}/test_day_corrected_alt_sza.npy'

data = np.load(data_file)

# If using original dataset without corrections. 
data[1] = data[1] * 85000

# Remove upper stratosphere.
data = data[:, data[9] > 20]
# Remove night.
data = data[:, data[12] > 0]

input_i = [0,1,2,3,4,5,8,9,10,11,12,16] # Original.
target_i = np.arange(18, 88) # All.
#target_i = 71 # H2O.
#target_i = 69 # NO3.
#target_i = 19 # NO2.
#target_i = 18 # O3.

# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, input_i, target_i)

# 90/10 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets)

# Make the regression model.
model = RandomForestRegressor(n_estimators=20, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)
model.fit(in_train, out_train)
preds = model.predict(in_test)

# Loop through each column and get the average differences.
grid = fns.make_cols_map(in_test, out_test, preds)
# Plot a map.
name = 'all' 

alt_bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000]

# For each alt level...
for i in range(len(alt_bins)-1):
  bottom = alt_bins[i]
  top = alt_bins[i+1]
  # Select the data at this level.
  bin_idx = np.where((in_test[:, 1] >= bottom) & (in_test[:, 1] <= top))[0]
  input, target, pred = in_test[bin_idx], out_test[bin_idx], preds[bin_idx]
  # Loop through each column and get the average differences.
  grid = fns.make_cols_map(input, target, pred) 
  # Get the average R2 score.
  target = target.ravel()
  pred = pred.ravel()
  r2 = round(r2_score(target, pred), 3) 
  print(r2)  
  name = f'all' 
  fns.show_diff_map(name, grid, r2)

