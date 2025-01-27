'''
Name: Sophie Turner.
Date: 16/1/2025.
Contact: st838@cam.ac.uk.
Plot column performance on a map.
'''
import os
import numpy as np
import constants as con
import cartopy.crs as ccrs
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm

# File paths.
mod_name = 'rf'
mod_path = f'{paths.mod}/{mod_name}/{mod_name}'
inputs_path = f'{mod_path}_test_inputs.npy' 
targets_path = f'{mod_path}_test_targets.npy'
preds_path = f'{mod_path}_pred.npy'
#grid_path = f'{paths.npy}/cols_map_NO3_alltimes.npy'
grid_path = f'{paths.npy}/cols_map_allJs_alltimes.npy'
grid_path = 'test'

# Load data.
print('\nLoading data.')
inputs = np.load(inputs_path)
targets = np.load(targets_path)
preds = np.load(preds_path)
print('Inputs:', inputs.shape)
print('Targets:', targets.shape)
print('Preds:', preds.shape)

if os.path.exists(grid_path):
  grid = np.load(grid_path)
else:
  # Takes about 200 minutes.
  # Create an array of column errors.
  print('Creating array of column errors.')
  # Get the unique lat & lon values.
  lats = inputs[:, 2]
  lats = np.unique(lats)
  lons = inputs[:, 3]
  lons = np.unique(lons)

  # Make a 2d list of coords, R2 scores and % differences for each lat and lon point.
  grid = []

  # For each lat...
  for i in range(len(lats)):
    lat = lats[i]
    print(f'Processing slice {i+1} of {len(lats)}')
    print('grid:', len(grid))
    # And each lon...
    for j in range(len(lons)):
      lon = lons[j]
      # Get the indices of all the target & pred J rates in that column.
      idx = np.where((inputs[:, 2] == lat) & (inputs[:, 3] == lon))
      # Get all the target J rates in that column.
      target = targets[idx] # J core.
      #target = targets[idx, 11].squeeze() # NO3.
      # There might not be a data point at this location, or only 1. Skip if so.
      if len(target) < 2:
        #print('Skipping a missing data point')
        continue
      # Get the predicted J rates.
      pred = preds[idx] # J core.
      #pred = preds[idx, 11].squeeze() # NO3.
      # Get the R2 score and % diff.
      r2 = r2_score(target, pred)
      diff = ((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100
      # Save the R2 scores in a 2d list for every lat & lon.
      grid.append([lat, lon, r2, diff])
      
  grid = np.array(grid)
  # Save the r2 array because it takes ages to make.
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
  c = grid[:, 2]
elif show == 'diff':
  cmap = LinearSegmentedColormap.from_list("Cdiff", ["darkblue", "blue", "deepskyblue", "cyan", "lawngreen", "yellow", "orange", "red", "firebrick"]) 
  vmin, vmax = -10, 10
  text = '% difference of J rate predictions to targets'
  c = grid[:, 3]

# Plot the metrics by lat & lon coords on the cartopy mollweide map.
ax = plt.axes(projection=ccrs.Mollweide())
plt.title(f'Column photolysis rates predicted by random forest') # NO{con.sub3}
plt.scatter(grid[:, 1], grid[:, 0], c=c, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), s=3)
plt.colorbar(shrink=0.5, label=f'{text} in column', orientation='horizontal')
ax.coastlines()
plt.show()
plt.close()
