'''
Name: Sophie Turner.
Date: 16/1/2025.
Contact: st838@cam.ac.uk.
Plot column performance on a map.
'''
import os
import numpy as np
import functions as fns
import constants as con
import cartopy.crs as ccrs
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Choose which J rate to look at.
j_idx, j_name = None, 'all' # Average over all J rates in J core.
#j_idx, j_name = 11, 'NO3'
# Choose whether to show R2 or % diff on map ('r2' or 'diff').
show = 'r2'

# Load trained random forest data.
inputs, targets, preds = fns.load_model_data('rf')

# Fetch or make the data for the map.
grid_path = f'{paths.npy}/cols_map_{j_name}_alltimes.npy'
if os.path.exists(grid_path):
  grid = np.load(grid_path)
else:
  # Takes up to 200 minutes.
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
      idx = idx[0]
      # Get all the target J rates in that column.
      target = targets[idx, j_idx].squeeze() 
      # There might not be a data point at this location, or only 1. Skip if so.
      if len(target) < 2:
        continue
      # Get the predicted J rates.
      pred = preds[idx, j_idx].squeeze() 
      # Get the R2 score and % diff.
      r2 = r2_score(target, pred)
      diff = ((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100
      # Save the R2 scores in a 2d list for every lat & lon.
      grid.append([lat, lon, r2, diff])
      
  grid = np.array(grid)
  # Save the r2 array because it takes ages to make.
  np.save(grid_path, grid)    
    
print(grid.shape)

# Colourmaps designed specifically for these data.
if show == 'r2':
  # Set negative R2s to zero to simplify the plot.
  grid[grid[:, 2] < 0, 2] = 0
  # Create a colourmap to represent the data better than the default ones.
  cmap = con.cmap_r2  
  vmin, vmax = 0, 1
  text = f'R{con.sup2} score of J rate predictions'
  c = grid[:, 2]
elif show == 'diff':
  cmap = con.cmap_diff
  vmin, vmax = -10, 10
  text = '% difference of J rate predictions to targets'
  c = grid[:, 3]

# Plot the metrics by lat & lon coords on the cartopy mollweide map.
ax = plt.axes(projection=ccrs.Mollweide())
plt.title(f'Column {j_name} photolysis rates predicted by random forest')
plt.scatter(grid[:, 1], grid[:, 0], c=c, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), s=3)
plt.colorbar(shrink=0.5, label=f'{text} in column', orientation='horizontal')
ax.coastlines()
plt.show()
plt.close()
