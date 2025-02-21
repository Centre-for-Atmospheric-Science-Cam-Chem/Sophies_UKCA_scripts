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
j_idx = 11
j_name = con.J_names[j_idx] 
j_name_s = con.J_names_short[j_idx] 
j_idx, j_name, j_name_s = None, 'all', 'all' # Average over all J rates in J core.

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
times = inputs[:, con.days]
times = np.unique(times)

# Set start date and time.
date = datetime(2015, 1, 1, 1)
date += timedelta(hours=start) 
  
# For each timestep...
for t in range(start, stop):
  
  time = times[t]
  print(f'Mapping timestep {date}.') 
  
  # Get the data in this timestep.
  idx = np.where(inputs[:, con.days] == time)
  targett = targets[idx]
  predt = preds[idx]
  inputt = inputs[idx]
  
  # Get the R2 score for this timestep.
  r2 = r2_score(targett, predt)
  
  # Make grid of cols. Returns 2d numpy array of lat, lon, r2, diff for each col.
  grid = fns.make_cols_map(inputt, targett, predt, j_idx)

  cmap = con.cmap_diff
  # Clip diffs to bounds to simplify the plot.
  vmin, vmax = -20, 20
  grid[grid[:, 3] < vmin, 3] = vmin
  grid[grid[:, 3] > vmax, 3] = vmax
  x = grid[:, 1]
  y = grid[:, 0]
  c = grid[:, 3]

  # Set the fig to a consistent size.
  plt.figure(figsize=(10,6))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide())
  plt.title(f'Columns of {j_name} photolysis rates predicted by random forest\n{date.strftime("%d/%m/%Y %H:%M")}  R{con.sup2} = {r2:.3f}')
  plt.scatter(x, y, s=7, c=c, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
  cbar = plt.colorbar(shrink=0.7, label=f'% difference of J rate predictions to targets in column', orientation='horizontal')
  cbar.ax.set_xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
  cbar.ax.set_xticklabels(['< -20', '-15', '-10', '-5', '0', '5', '10', '15', '> 20'])
  ax.set_global()
  ax.coastlines()
  
  # Test.
  #plt.show()
  #exit()

  # Save the fig.
  fig_name = date.strftime('%d%H')
  map_path = f'{dir_path}/{fig_name}.png'
  plt.savefig(map_path)
  plt.close()

  # Add time to start date.
  date += timedelta(hours=1)  
 
# Turn the pictures into a GIF.
fns.make_gif(dir_path, gif_name=dir_name)
