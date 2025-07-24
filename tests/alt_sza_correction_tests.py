'''
Name: Sophie Turner.
Date: 21/5/2025.
Contact: st838@cam.ac.uk.
Compare performace of random forests with and without corrected altitude and SZA
at different altitudes.
Not for use with my standard .npy data before May 2025.
'''

import time
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#print('\nOriginal dataset without corrections.')
#data_file = f'{paths.npy}/test_day.npy'

print('\nDataset with the correct altitudes and scaled SZA.')
data_file = f'{paths.npy}/test_day_corrected_alt_sza.npy'

data = np.load(data_file)

# Remove upper stratosphere.
data = data[:, data[9] > 20]
# Remove night.
data = data[:, data[12] > 0]

#input_i = [0,1,2,3,4,5,8,9,10,16] # Original.
#input_i = np.arange(18) # All.
input_i = [0,1,2,3,4,5,6,8,9,10,11,12,15,16,17] # All except orography & longwave fluxes.
#target_i = np.arange(18, 88) # All.
target_i = 71 # H2O.

# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, input_i, target_i)

# 90/10 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets)

# Make the regression model.
model = RandomForestRegressor(n_estimators=20, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)
model.fit(in_train, out_train)
preds = model.predict(in_test)

# Get the unique lat & lon values.
lats = inputs[:, con.lat]
lats = np.unique(lats)
lons = inputs[:, con.lon]
lons = np.unique(lons)

# Make a 2d list of coords, R2 scores and % differences for each lat and lon point.
grid = []
# For each lat...
for i in range(len(lats)):
  lat = lats[i]
  # And each lon...
  for j in range(len(lons)):
    lon = lons[j]
    # Get the indices of all the data in that column.
    idx = np.where((in_test[:, con.lat] == lat) & (in_test[:, con.lon] == lon))[0]
    # Get all the target J rates in that column.
    target = out_test[idx].squeeze()     
    # There might not be a data point at this location. Skip if so.
    if not np.any(target) or target.ndim == 0:
      continue
    # Get the predicted J rates in the column.
    pred = preds[idx].squeeze() 
    # Get the R2 score and % diff.
    r2 = r2_score(target, pred)
    diff = np.nan_to_num(((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100, posinf=0, neginf=0)
    # Save the R2 scores in a 2d list for every lat & lon.
    grid.append([lat, lon, r2, diff])     
grid = np.array(grid)

r2 = r2_score(out_test, preds)
mse = mean_squared_error(out_test, preds)
print(f'Random forest trained for H2O only, with all inputs including O3 col.\nR2 = {r2}\nMSE = {mse}')

fns.show_diff_map('water', grid, r2)
