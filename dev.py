'''
Name: Sophie Turner.
Date: 11/2/2025.
Contact: st838@cam.ac.uk.
Simulate the simulation! Actually, simulate the emulation of the simulation!!
Assume the UM provides the random forest with inputs for only the current timestep.
Test giving the random forest the whole spatial grid at once.
Test giving the random forest one column at a time.
Compare accuracy. Timings are for info only, not indicative of the final implementation.
'''

import os
import time
import joblib
import warnings
import numpy as np
import functions as fns
import constants as con
import cartopy.crs as ccrs
import file_paths as paths
from idx_names import idx_names
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def show_map(fullname, grid, r2, out_path):
  '''Show a map of the column % diffs for a specific J rate.
  fullname: string, fully formatted name of J rate.
  grid: 2d numpy array of lat, lon, r2, diff for each column.
  r2: number, overall R2 score of whole grid at this timestep.
  out_path: string, file path to save fig as.
  '''
  # Show plot for this J rate and ts. 
  cmap = con.cmap_diff
  vmin, vmax = -20, 20
  # Clip the bounds to +- 20% to remove ridiculous outliers and simplify the plot visuals.
  grid[grid[:, 3] < vmin, 3] = vmin
  grid[grid[:, 3] > vmax, 3] = vmax
  x = grid[:, 1]
  y = grid[:, 0]
  c = grid[:, 3]
  # Set the fig to a consistent size.
  plt.figure(figsize=(10,7.5))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide()) 
  plt.title(f'Columns of {fullname} photolysis rates predicted by random forest. Overall {con.r2} = {r2}')
  plt.scatter(x, y, c=c, s=3, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree()) 
  plt.colorbar(shrink=0.5, label=f'% difference of J rate predictions to targets in column', orientation='horizontal')
  ax.set_global()
  ax.coastlines()
  plt.show()
  # Save the fig.
  plt.savefig(out_path)
  plt.close()


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

'''
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

  # Get performance metrics for this J rate for this column. 
  grid = fns.make_cols_map(inputs, targets, preds, i)
  
  # Get total R2 for this J rate.
  target = targets[:, i]
  pred = preds[:, i]
  r2 = round(r2_score(target, pred), 3)
  print(f'Overall R2 for {fullname} in whole grid at this timestep: {r2}') 

  # Show plot for this J rate and ts. 
  map_path = f'{paths.analysis}/col_maps_full_res/{shortname}_whole_grid.png'
  show_map(fullname, grid, r2, map_path)

# Record time taken.
end = time.time()
sec = end - start
print(f'It took {round(sec/60)} minutes in total, to make & save maps of every J rate in this timestep using the whole grid.')
'''

# Test giving the random forest single columns at a time as if it was in UKCA.
# If it's worse, inputs from previous timesteps could be included at little extra cost. This would mean the first few sim hours should be discarded from science like in spin-up.
# The reason things are done in a weird and long-winded order here is to try and replicate how the steps will have to be done in UKCA run-time.
print('\nUsing random forest on individual columns.')
start = time.time()
# Select inputs and targets.
inputs, targets = fns.in_out_swap(data, con.phys_main, con.J_all)
# Make an array to put all the preds in. Start with the targets array just because it's the right shape and will be indexed the same.
preds, truths = np.full(targets.shape, np.nan), np.full(targets.shape, np.nan)

# Pick out each column.
# Get the unique lat & lon values.
lats = inputs[:, con.lat]
lats = np.unique(lats)
lons = inputs[:, con.lon]
lons = np.unique(lons)
# Make a 2d list of coords and % differences for each lat and lon point.
grid = []  
# Ignore warnings about div by zero because we've caught that.
warnings.filterwarnings('ignore') 

# For each lat...
for i in range(len(lats)):
  lat = lats[i]
  # And each lon...
  for j in range(len(lons)):
    lon = lons[j]
    
    # Get the indices of this column.
    idx = np.where((inputs[:, con.lat] == lat) & (inputs[:, con.lon] == lon))[0]   
    # Get the inputs in this column.
    input = inputs[idx]
    # Skip if the column is at night.
    if np.all(input[:, con.down_sw_flux] == 0):
      continue
    # Get the targets in the column.
    target = targets[idx]
        
    # Select the pressures within range of fast-J.
    trop = np.where(input[:, con.pressure] > 20)[0]
    input = input[trop].squeeze()
    target = target[trop].squeeze()   
    # Keep these indices for later.
    idx = idx[trop]
    
    # Use the trained (not scaled) RF on the inputs for this column only.
    pred = rf.predict(input)
    
    # Store these preds in the full preds array.
    preds[idx] = pred
    truths[idx] = target
    
    # Get overall % diff for this column.
    diff = np.nan_to_num(((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100, nan=np.nan, posinf=0, neginf=0)
    
    # Place preds and % diff in an array for the full grid. Pad 2nd index to keep to same structure as other grid, for use in fns.
    grid.append([lat, lon, 0, diff])
    '''
    # Get % diffs for each J rate.
    for k in range(len(pred[0])):
      pred_rate = pred[:, k]
      target_rate = target[:, k]
      # Get % diff for this J rate in this column.
      diff = np.nan_to_num(((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100, nan=np.nan, posinf=0, neginf=0)
    '''
    
# Record time taken.
end = time.time()
sec = end - start
print(f'Using the random forest on each individual column in the whole grid at one timestep took {round(sec)} seconds.')
grid = np.array(grid)
print('grid:', grid.shape)

# Calculate R2 of whole grid at this timestep. Remove nans first.
truths, preds = truths[~np.isnan(truths)], preds[~np.isnan(preds)] 
r2 = round(r2_score(truths, preds), 3)
print(f'\nOverall R2 for all J rates at each column in timestep: {r2}\n')

# Show plots for all combined J rates.
map_path = f'{paths.analysis}/col_maps_full_res/all_individual_cols.png'
show_map('all', grid, r2, map_path)


