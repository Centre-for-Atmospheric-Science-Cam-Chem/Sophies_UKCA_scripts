'''
Name: Sophie Turner.
Date: 11/2/2025.
Contact: st838@cam.ac.uk.
Simulate the simulation! Actually, simulate the emulation of the simulation!!
Assume the UM provides the random forest with inputs for only the current timestep.
Test giving the random forest the whole spatial grid at once.
Compare accuracy. Timings are for info only, not indicative of the final implementation.
'''

import time
import joblib
import numpy as np
import functions as fns
import constants as con
import file_paths as paths
from idx_names import idx_names
from sklearn.metrics import r2_score

# File paths.
day_path = f'{paths.npy}/20160401.npy'
rf_path = f'{paths.mod}/rf/rf.pkl'

# Load a full day of np data (ALL J rates and full resolution).
print('\nLoading data.')
data = np.load(day_path)
print('Data:', data.shape)

# Load the trained (not scaled) random forest.
rf = joblib.load(rf_path) 
# Pick out 1 timestep.
data = data[:, data[con.hour] == 12] # Midday.
print('Timestep:', data.shape)

# Test giving the random forest the whole grid at once (for 1 ts) as if it was in UKCA.
print('\nUsing random forest on whole timestep.')
start = time.time()
# Ignore night and upper strat.
data = fns.day_trop(data)
# Select inputs and targets.
inputs, targets = fns.in_out_swap(data, con.phys_main, con.J_all)

# Use the random forest on them.
preds = rf.predict(inputs)
# Calculate R2 of whole grid at this timestep.
r2 = round(r2_score(targets, preds), 3)
# Record time taken.
end = time.time()
sec = end - start
print(f'Using the random forest on the whole grid at one timestep took {round(sec)} seconds.')
print(f'\nOverall R2 for all J rates and whole grid in timestep: {r2}\n')

# Get performance metrics for this J rate for each column and save array of performance per col.
grid = fns.make_cols_map(inputs, targets, preds, None)
# Plot it on the map.
map_path = f'{paths.analysis}/col_maps_full_res/all_whole_grid.png'
show_map('all', grid, r2, map_path)

# Now do the grid and map for each J rate individually.
for i in range(len(targets[0])):  
  # Skip if this is an empty output.
  if np.all(targets[:, i] == 0):          
    continue
  
  # Get the name of the reaction.
  shortname = idx_names[i + con.n_phys][3]
  fullname = idx_names[i + con.n_phys][2]

  # Get performance metrics for this J rate for each column. 
  grid = fns.make_cols_map(inputs, targets, preds, i)
  
  # Get total R2 for this J rate.
  target = targets[:, i]
  pred = preds[:, i]
  r2 = round(r2_score(target, pred), 3)
  print(f'Overall R2 for {fullname} in whole grid at this timestep: {r2}') 

  # Show plot for this J rate and ts. 
  map_path = f'{paths.analysis}/col_maps_full_res/{shortname}_whole_grid.png'
  fns.show_diff_map(fullname, grid, r2, map_path)

# Record time taken.
end = time.time()
sec = end - start
print(f'It took {round(sec/60)} minutes in total, to make & save maps of every J rate in this timestep using the whole grid.')
