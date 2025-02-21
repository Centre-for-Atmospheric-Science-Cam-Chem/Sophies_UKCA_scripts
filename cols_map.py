'''
Name: Sophie Turner.
Date: 16/1/2025.
Contact: st838@cam.ac.uk.
Plot column performance on a map.
'''
import os
import warnings
import numpy as np
import functions as fns
import constants as con
import cartopy.crs as ccrs
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Pick which J rate to look at.
j_idx = 0
j_name = con.J_names[j_idx] 
j_name_s = con.J_names_short[j_idx] 
#j_idx, j_name, j_name_s = None, 'all', 'all' # Average over all J rates in J core.

# Choose whether to show R2 or % diff on map ('r2' or 'diff').
show = 'r2'

# Fetch or make the data for the map.
grid_path = f'{paths.npy}/cols_map_{j_name_s}.npy'
if os.path.exists(grid_path):
  grid = np.load(grid_path)
else:
  # Load trained random forest data.
  inputs, targets, preds = fns.load_model_data('rf')
  # Create the grid of column performaces.
  print('Creating the map grid of column performace. This could take up to 2 hours or even longer!')
  grid = fns.make_cols_map(inputs, targets, preds, j_idx, grid_path)

if show == 'r2':
  # Set negative R2s to zero to simplify the plot.
  grid[grid[:, 2] < 0, 2] = 0
  # Colourmaps designed specifically for these data.
  cmap = con.cmap_r2  
  vmin, vmax = 0.7, 1
  text = f'R{con.sup2} score of J rate predictions'
  c = grid[:, 2]
elif show == 'diff':
  cmap = con.cmap_diff
  vmin, vmax = -10, 10
  # Clip the bounds to +- 10% to remove ridiculous outliers and simplify the plot visuals.
  grid[grid[:, 3] < vmin, 3] = vmin
  grid[grid[:, 3] > vmax, 3] = vmax
  text = '% difference of J rate predictions to targets'
  c = grid[:, 3]

# Plot the metrics by lat & lon coords on the cartopy mollweide map.
ax = plt.axes(projection=ccrs.Mollweide())
plt.title(f'Columns of {j_name} photolysis rates predicted by random forest')
plt.scatter(grid[:, 1], grid[:, 0], c=c, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), s=3) 
plt.colorbar(shrink=0.5, label=f'{text} in column', orientation='horizontal')
ax.coastlines()
plt.show()
plt.close()
