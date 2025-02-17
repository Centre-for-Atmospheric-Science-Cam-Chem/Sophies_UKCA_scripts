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
from datetime import datetime, timedelta

# Pick which J rate to look at.
j_idx = 0
j_name = con.J_names_short[j_idx] 
j_idx, j_name = None, 'all' # Average over all J rates in J core.

# No need to show the warning beacuse it's been caught.
warnings.simplefilter('ignore')

# Get ready to save figs.
dir_name = f'col_maps_year_{j_name}_scaled'
dir_path = f'{paths.analysis}/{dir_name}'
if not os.path.exists(dir_path):
  print(f'\nMaking a new directory: {dir_path}')
  os.mkdir(dir_path)

# Load trained random forest data.
inputs, targets, preds = fns.load_model_data('rf_scaled')
 
# Select daily timesteps for the year.
times = inputs[:, 4]
times = np.round(times)
inputs[:, 4] = times
times = np.unique(times)

# Set start date.
date = datetime(2015, 1, 1)
  
# For each timestep...
for t in range(len(times)):
  time = int(times[t])
  print(f'Mapping timestep {t+1} of {len(times)}.')  
  
  # Get the data in this timestep.
  idx = np.where(inputs[:, 4] == time)
  targett = targets[idx]
  predt = preds[idx]
  inputt = inputs[idx]
  
  # Get the R2 score for this timestep.
  r2 = round(r2_score(targett, predt), 3)
  
  # Make the grid of cols. Returns 2d np array of lat, lon, r2, diff for each col.
  grid = fns.make_cols_map(inputt, targett, predt, j_idx)
  
  cmap = con.cmap_diff  
  vmin, vmax = -20, 20
  # Clip diffs to bounds to simplify the plot.
  grid[grid[:, 3] < vmin, 3] = vmin
  grid[grid[:, 3] > vmax, 3] = vmax
  x = grid[:, 1]
  y = grid[:, 0]
  c = grid[:, 3]

  # Set the fig to a consistent size.
  plt.figure(figsize=(10,6))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide())
  plt.title(f'Columns of {j_name} photolysis rates predicted by random forest\n{date.strftime("%d/%m/%Y")}  R{con.sup2} = {r2}')
  plt.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), marker='s', s=2)
  cbar = plt.colorbar(shrink=0.7, label=f'% difference of J rate predictions to targets in column', orientation='horizontal')
  cbar.ax.set_xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
  cbar.ax.set_xticklabels(['< -20', '-15', '-10', '-5', '0', '5', '10', '15', '> 20'])
  ax.set_global()
  ax.coastlines()

  # Prepend zeros to file name digits so that the GIF goes in the right order.
  if time < 10:
    fig_name = f'00{time}'
  elif time < 100:
    fig_name = f'0{time}'
  else:
    fig_name = time

  # Save the fig.
  map_path = f'{dir_path}/{fig_name}.png'
  plt.savefig(map_path)
  plt.close()

  # Add time to start date.
  date += timedelta(days=1)
  
# Turn the pictures into a GIF.
fns.make_gif(dir_path, gif_name=dir_name, ms=250)
