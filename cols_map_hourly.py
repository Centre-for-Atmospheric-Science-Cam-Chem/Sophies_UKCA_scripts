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
from datetime import datetime, timedelta

# Pick which J rate to look at.
j_idx = 20
j_name = con.J_names[j_idx] 
j_name_s = con.J_names_short[j_idx] 
#j_idx, j_name, j_name_s = None, 'all', 'all' # Average over all J rates in J core.

# Pick a time to look at.
# First and last hours of 3 days.
spring = [1847, 1919, 'spring'] # A day each side of the equinox.
summer = [4079, 4151, 'summer'] # A day each side of the solstice.
autumn = [6335, 6407, 'autumn'] # A day each side of the equinox.
winter = [8495, 8567, 'winter'] # A day each side of the solstice.
season = autumn
start, stop, season = season[0], season[1], season[2]

# No need to show the warning beacuse it's been caught.
warnings.simplefilter('ignore')

# Get ready to save figs.
dir_name = f'col_maps_hourly_{j_name_s}'
dir_path = f'{paths.analysis}/{dir_name}'
if not os.path.exists(dir_path):
  print(f'\nMaking a new directory: {dir_path}')
  os.mkdir(dir_path)

# Load trained random forest data.
inputs, targets, preds = fns.load_model_data('rf')
 
# Select daily timesteps for the year.
times = inputs[:, 4]
times = np.unique(times)

# Set start date and time.
date = datetime(2015, 1, 1, 1)
date += timedelta(hours=start) 
  
# For each timestep...
for t in range(start, stop):
  
  time = times[t]
  print(f'Mapping timestep {date}.')  
  
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

  # Make a 2d list of coords and % differences for each lat and lon point.
  grid = []

  # For each lat...
  for i in range(len(lats)):
    lat = lats[i]
    # And each lon...
    for j in range(len(lons)):
      lon = lons[j]
      # Get the indices of all the target & pred J rates in that column.
      idx = np.where((inputt[:, 2] == lat) & (inputt[:, 3] == lon))
      idx = idx[0]
      # Get all the target J rates in that column.
      target = targett[idx, j_idx]
      # There might not be a data point at this location. Skip if so.
      if len(target) < 1:
        continue
      # Get the predicted J rates.
      pred = predt[idx, j_idx] 
      # Get the % diff.
      diff = np.nan_to_num(((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100, nan=np.nan, posinf=0, neginf=0)
      # Save them in a 2d list for every lat & lon.
      grid.append([lat, lon, diff])

  grid = np.array(grid)   

  cmap = con.cmap_diff
  vmin, vmax = -20, 20
  # Clip diffs to bounds to simplify the plot.
  grid[grid[:, 2] < vmin, 2] = vmin
  grid[grid[:, 2] > vmax, 2] = vmax
  text = '% difference of J rate predictions to targets'
  x = grid[:, 1]
  y = grid[:, 0]
  c = grid[:, 2]

  # Set the fig to a consistent size.
  plt.figure(figsize=(10,6))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide())
  plt.title(f'Columns of {j_name} photolysis rates predicted by random forest\n{date.strftime("%d/%m/%Y %H:%M")}')
  plt.scatter(x, y, s=7, c=c, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
  cbar = plt.colorbar(shrink=0.7, label=f'{text} in column', orientation='horizontal')
  cbar.ax.set_xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
  cbar.ax.set_xticklabels(['< -20', '-15', '-10', '-5', '0', '5', '10', '15', '> 20'])
  ax.set_global()
  ax.coastlines()

  # Save the fig.
  fig_name = date.strftime('%d%H')
  map_path = f'{dir_path}/{fig_name}.png'
  plt.savefig(map_path)
  plt.close()

  # Add time to start date.
  date += timedelta(hours=1)  
 
# Turn the pictures into a GIF.
fns.make_gif(dir_path, gif_name=dir_name)
