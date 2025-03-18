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
from idx_names import idx_names
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

# Make empty diff array, to append to results array for each col.
# It will be 2d with diff for each J rate.
diffs = []

# For each lat...
for i in range(len(lats)):
  lat = lats[i]
  # And each lon...
  for j in range(len(lons)):
    lon = lons[j]
    # Get the indices of all the data in this column.
    idx = np.where((inputs[:, con.lat] == lat) & (inputs[:, con.lon] == lon))[0]      
    # Get the inputs in this column.
    input = inputs[idx]
    # Skip if the column is at night.
    if np.all(input[:, con.down_sw_flux] == 0):
      continue
    # Get the target J rates in the column.
    target = targets[idx]
        
    # Select the pressures within range of fast-J.
    trop = np.where(input[:, con.pressure] > 20)[0]
    input = input[trop].squeeze()
    target = target[trop].squeeze()

    # Use the trained (not scaled) RF on the inputs for this column only.
    pred = rf.predict(input)

    # Get overall % diffs for all J rates in this column.
    diff = np.nan_to_num(((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100, nan=np.nan, posinf=0, neginf=0)
    # Store diff in the diffs array.
    diffs.append(diff)  

    # For each J rate...
    for r in range(n_J):
      target_J = target[:, r]
      pred_J = pred[:, r]
      # Get % diff for this J rate in this column.
      diff = np.nan_to_num(((np.mean(pred_J) - np.mean(target_J)) / np.mean(target_J)) * 100, nan=np.nan, posinf=0, neginf=0)     
      # Store diff in diffs array.
      diffs.append(diff)

    # Pad lat, lon and diffs so they fit nicely with the dims in results array.
    cols = target.shape[0]
    lat_res = np.full((rows, cols), lat)
    lon_res = np.full((rows, cols), lon)
    diffs_res = [diffs] * cols
    diffs_res = np.array(diffs_res)
    diffs_res = diffs_res.T    
    # Pad target and pred to account for the 0th index being the average.
    target_res = np.pad(target, ((0,0), (1,0)), mode='constant', constant_values=0)
    pred_res = np.pad(pred, ((0,0), (1,0)), mode='constant', constant_values=0)
    # Flip target and pred so they're in the same shape as the others.
    target_res, pred_res = target_res.T, pred_res.T
    
    # Add all this data to the full results array.
    # [lat, lon, target, pred, diffs].   
    results.append([lat_res, lon_res, target_res, pred_res, diffs_res])
    
    '''
    # TEST.    
    print('\ntarget', target.shape)
    print('pred', pred.shape)
    print('diffs', diffs.shape)
    print('lat', lat.shape)
    print('lon', lon.shape)
    results = np.array(results).squeeze()
    print('\nresults', results.shape)
    exit()
    '''
    
# Turn the results into a np array for easier operations.
results = np.array(results)
print(results.shape)

# For each J rate...
for r in range(n_J):
  # Get the name unless it's the average one of all. 
  # Pred and target data will be different too if it's the average of all.
  if r == 0:
    fullname, shortname = 'all', 'all'
    target = results[:, 2]
    pred = results[:, 3] 
  else:
    fullname = idx_names[r + 1 + con.n_phys][2]
    shortname = idx_names[r + 1 + con.n_phys][3]
    target = results[:, 2, r]
    pred = results[:, 3, r]
  
  # Get the rest of the data for this J rate.
  lat = results[:, 0, r] 
  lon = results[:, 1, r]
  diff = results[:, 4, r]
  
  # Get the overall R2 for this J rate.
  r2 = round(r2_score(target, pred), 3)  

  # Show lat-lon map of diffs for this J rate and time.
  map_path = f'{paths.analysis}/col_maps_full_res/{shortname}_individual_cols.png'
  cmap = con.cmap_diff
  # Clip the bounds to +- 20% to remove ridiculous outliers and simplify the plot visuals.
  vmin, vmax = -20, 20
  grid[grid[:, 4] < vmin, 3] = vmin
  grid[grid[:, 4] > vmax, 3] = vmax
  # Set the fig to a consistent size.
  plt.figure(figsize=(10,7.5))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide()) 
  plt.title(f'Columns of {fullname} photolysis rates predicted by random forest. Overall {con.r2} = {r2}')
  plt.scatter(lon, lat, c=diff, s=3, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree()) 
  plt.colorbar(shrink=0.5, label=f'% difference of J rate predictions to targets in column', orientation='horizontal')
  ax.set_global()
  ax.coastlines()
  
  # TEST.
  # Save the fig.
  #plt.savefig(out_path)
  plt.show()
  plt.close() 
  exit()


