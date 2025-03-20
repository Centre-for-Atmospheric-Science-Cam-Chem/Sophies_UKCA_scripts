'''
Name: Sophie Turner.
Date: 13/3/2025.
Contact: st838@cam.ac.uk.
Simulate the simulation! Actually, simulate the emulation of the simulation!!
Assume the UM provides the random forest with inputs for only the current timestep.
Test giving the random forest one column at a time.
Compare accuracy. Timings are for info only, not indicative of the final implementation.
'''

import time
import joblib
import warnings
import numpy as np
import functions as fns
import constants as con
import file_paths as paths
import cartopy.crs as ccrs
from idx_names import idx_names
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# File paths.
day_path = f'{paths.npy}/20160401.npy'
rf_path = f'{paths.mod}/rf_trop/rf_trop.pkl'

# Load a full day of np data (ALL J rates and full resolution).
print('\nLoading data.')
data = np.load(day_path)
print('Data:', data.shape)

# Load the trained (not scaled) random forest.
rf = joblib.load(rf_path) 
# Pick out 1 timestep.
data = data[:, data[con.hour] == 12] # Midday.
print('Timestep:', data.shape)

# Test giving the random forest single columns at a time as if it was in UKCA.
# If it's worse, inputs from previous timesteps could be included at little extra cost. This would mean the first few sim hours should be discarded from science like in spin-up.
# The reason things are done in a weird and long-winded order here is to try and replicate how the steps will have to be done in UKCA run-time.
print('\nUsing random forest on individual columns.')
# Select inputs and targets. 
inputs, targets = fns.in_out_swap(data, con.phys_main, con.J_trop)

# Pick out each column.
# Get the unique lat & lon values.
lats = inputs[:, con.lat]
lats = np.unique(lats)
lons = inputs[:, con.lon]
lons = np.unique(lons) 
# Ignore warnings about div by zero because we've caught that.
warnings.filterwarnings('ignore') 

# Get number of reactions and desired shapes of results.
n_J = len(con.J_trop)
rows = n_J + 1

# Make empty preds and targets array. This is so that we only save data where there are J rates and not a load of null space.
# It will have [lat, lon, target, pred, diffs] for each col.
results = [] 

# For each lat...
for i in range(len(lats)):
  lat = lats[i]  
  print(f'Processing latitude {i + 1} of {len(lats)}')
  # And each lon...
  for j in range(len(lons)):
    lon = lons[j]
    # Get the indices of all the data in this column.
    idx = np.where((inputs[:, con.lat] == lat) & (inputs[:, con.lon] == lon))[0]      
    # Get the inputs in this column.
    input = inputs[idx]
    
    # Select the lowest altitudes within range of fast-J.
    # Normally it would be pressure but altitude used here to allow 2D indexing later.
    # Needs to be done before checking for night or we get empty cols of perfect preds.
    trop = np.where(input[:, con.alt] < 0.62)[0]
    input = input[trop].squeeze()
    
    # Skip if the column is at night.
    if np.all(input[-1, con.down_sw_flux] == 0):          
      continue   
      
    # Get the target J rates in the column.
    target = targets[idx][trop].squeeze()

    # Use the trained (not scaled) RF on the inputs for this column only.
    pred = rf.predict(input)

    # Array of % differences for each J rate in this column. 
    diffs = [] 

    # For each J rate...
    for r in range(n_J):
      target_J = target[:, r]
      pred_J = pred[:, r]
      # Get % diff for this J rate in this column.
      diff = np.nan_to_num(((np.mean(pred_J) - np.mean(target_J)) / np.mean(target_J)) * 100, nan=np.nan, posinf=0, neginf=0)     
      # Store diff in diffs array.
      diffs.append(diff)
   
    # Get overall % diffs for all J rates in this column.
    diff = np.nan_to_num(((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100, nan=np.nan, posinf=0, neginf=0)
    # Stick it on the end.
    diffs.append(diff) 
   
    # Pad target and pred to account for the last index being the average.
    target = np.pad(target, ((0,0), (0,1)), mode='constant', constant_values=0)
    pred = np.pad(pred, ((0,0), (0,1)), mode='constant', constant_values=0)
    # Flip target and pred so they're in the same shape as the others.
    target, pred = target.T, pred.T
      
    # Add all this data to the full results array.
    # [lat, lon, target, pred, diffs].   
    results.append([lat, lon, target, pred, diffs])  
    
# Keep track of the R2 scores to get an average after.
r2_sum = 0
# For each J rate...
for r in range(n_J + 1):
  # Get the name unless it's the average one of all (at the last index). 
  # Pred and target data will be different too if it's the average of all.
  if r == n_J:
    fullname, shortname = 'all', 'all'
    r2 = round(r2_sum / n_J, 3)
  else:
    fullname = idx_names[r + con.n_phys][2]
    shortname = idx_names[r + con.n_phys][3]
    target = [row[2][r] for row in results] # target = results[:, 2, r]
    pred = [row[3][r] for row in results] # pred = results[:, 3, r]
    # Flatten.
    target = [item for sublist in target for item in sublist]
    pred = [item for sublist in pred for item in sublist]
    
    # Get the overall R2 for this J rate.
    r2 = round(r2_score(target, pred), 3)
    # Keep track of the R2 scores to get an average after.
    r2_sum += r2 
  
  # Get the rest of the data for this J rate.
  lat = [row[0] for row in results] # lat = results[:, 0, r] 
  lon = [row[1] for row in results] # lon = results[:, 1, r]
  diff = [row[4][r] for row in results] # diff = results[:, 4, r]

  # Show lat-lon map of diffs for this J rate and time.
  map_path = f'{paths.analysis}/col_maps_full_res/{shortname}_individual_cols.png'
  cmap = con.cmap_diff
  # Clip the bounds to +- 20% to remove ridiculous outliers and simplify the plot visuals.
  vmin, vmax = -20, 20
  # Set the fig to a consistent size.
  plt.figure(figsize=(10,7.5))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide()) 
  plt.title(f'Columns of {fullname} photolysis rates predicted by random forest. Overall {con.r2} = {r2}')
  plt.scatter(lon, lat, c=diff, s=3, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree()) 
  plt.colorbar(shrink=0.5, label=f'% difference of J rate predictions to targets in column', orientation='horizontal')
  ax.set_global()
  ax.coastlines()
  
  # Save the fig.
  plt.savefig(out_path)
  plt.show()
  plt.close() 
