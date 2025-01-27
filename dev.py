'''
Name: Sophie Turner.
Date: 16/1/2025.
Contact: st838@cam.ac.uk.
Plot column performance on a map.
'''
import os
import warnings
import numpy as np
import constants as con
import cartopy.crs as ccrs
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

# Ignore warnings about r2 with 1 sample. It doesn't matter.
warnings.simplefilter('ignore')

# File paths.
mod_name = 'rf'
mod_path = f'{paths.mod}/{mod_name}/{mod_name}'
inputs_path = f'{mod_path}_test_inputs.npy' 
targets_path = f'{mod_path}_test_targets.npy'
preds_path = f'{mod_path}_pred.npy'

# Load data.
print('\nLoading data.')
inputs = np.load(inputs_path)
targets = np.load(targets_path)
preds = np.load(preds_path)
print('Inputs:', inputs.shape)
print('Targets:', targets.shape)
print('Preds:', preds.shape)
 
# Select daily timesteps for the year.
times = inputs[:, 4]
times = np.round(times)
inputs[:, 4] = times
times = np.unique(times)

# Set start date.
date = datetime(2015, 1, 1)
  
# For each timestep...
for t in range(len(times)):
  time = times[t]
  print(f'Mapping timestep {t+1} of {len(times)}.')  
  
  # Get the data in this timestep.
  idx = np.where(inputs[:, 4] == time)
  targett = targets[idx]
  predt = preds[idx]
  inputt = inputs[idx]
  
  # Get the unique lat & lon values.
  lats = inputt[:, 2]
  lats = np.unique(lats)
  lons = inputt[:, 3]
  lons = np.unique(lons) 

  # Make a 2d list of coords, R2 scores and % differences for each lat and lon point.
  grid = []

  # For each lat...
  for i in range(len(lats)):
    lat = lats[i]
    # And each lon...
    for j in range(len(lons)):
      lon = lons[j]
      # Get the indices of all the target & pred J rates in that column.
      idx = np.where((inputt[:, 2] == lat) & (inputt[:, 3] == lon))
      # Get all the target J rates in that column.
      target = targett[idx] # J core.
      #target = targets[idx, 11] # NO3.
      # There might not be a data point at this location. Skip if so.
      if len(target) < 1:
        continue
      # Get the predicted J rates.
      pred = predt[idx] # J core.      
      #pred = preds[idx, 11] # NO3.
      # Get the R2 score and % diff.
      # R2 doesn't work properly with too few samples.
      if len(target) < 10:
        r2 = np.nan
      else:
        r2 = r2_score(target, pred)
      diff = ((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100
      # Save the R2 scores in a 2d list for every lat & lon.
      grid.append([lat, lon, r2, diff])

  grid = np.array(grid)
  # Save the array because it takes ages to make.
  grid_path = f'{paths.npy}/col_maps_year_allJs/{time}.npy' 
  np.save(grid_path, grid)    
    
  print(grid.shape)

  # Set negative R2s to zero to simplify the plot.
  grid[grid[:, 2] < 0, 2] = 0

  # Whether to show R2 or % diff on map.
  show = 'r2'

  # Colourmaps designed specifically for these data.
  if show == 'r2':
    # Create a colourmap to represent the data better than the default ones.
    cmap = LinearSegmentedColormap.from_list("Cr2", ["black", "black", "maroon", "darkred", "firebrick", "red", "crimson", "deeppink", "hotpink", "violet", "fuchsia", "orchid", "mediumorchid", "darkorchid", \
                                             "blueviolet", "mediumslateblue", "blue", "royalblue", "cornflowerblue", "dodgerblue", "deepskyblue", "darkturquoise", "turquoise", "cyan", "aquamarine", \
	       				     "mediumspringgreen", "lime", "limegreen", "forestgreen"])
    vmin, vmax = 0, 1
    text = f'R{con.sup2} score of J rate predictions'
    # Remove nans.    
    idx = ~np.isnan(grid[:, 2])    
    x = grid[idx, 1]
    y = grid[idx, 0]
    c = grid[idx, 2]

  elif show == 'diff':
    cmap = LinearSegmentedColormap.from_list("Cdiff", ["darkblue", "blue", "deepskyblue", "cyan", "lawngreen", "yellow", "orange", "red", "firebrick"]) 
    vmin, vmax = -10, 10
    text = '% difference of J rate predictions to targets'
    x = grid[:, 1]
    y = grid[:, 0]
    c = grid[:, 3]

  # Set the fig to a consistent size.
  plt.figure(figsize=(10,6))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide())
  plt.title(f'Column photolysis rates predicted by random forest\n{date.strftime("%d/%m/%y")}')
  plt.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), s=2)
  cbar = plt.colorbar(shrink=0.7, label=f'{text} in column', orientation='horizontal')
  if show == 'diff':
    cbar.ax.set_xticks([-10, -5, 0, 5, 10])
    cbar.ax.set_xticklabels(['-10', '-5', '0', '5', '10'])
  ax.coastlines()
  plt.show()

  # Save the fig.
  map_path = f'{paths.analysis}/col_maps_year_allJs/{time}.png'
  plt.savefig(map_path)
  plt.close()

  # Add time to start date.
  date += timedelta(days=1)
