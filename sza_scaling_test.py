'''
Name: Sophie Turner.
Date: 31/3/2025.
Contact: st838@cam.ac.uk.
Compare random forests with and without scaling solar zenith angle with vertical height.
Found no significant change from scaling SZA.
'''

import numpy as np
import functions as fns
import constants as con
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def compare_sza_scaling(model_name, title):
  '''Look at performance of random forest trained of surface level, 
  vertically scaled or pressure scaled solar zenith angle (SZA).
  model_name (string): the name of the random forest to use as it is named in its directory.
  title (string): description of type of SZA scaling used.
  '''
  # Load model data.
  inputs, targets, preds = fns.load_model_data(model_name)
  # Get R2 of all.
  r2 = r2_score(targets, preds)
  print(f'R{con.sup2} with {title}: {round(r2, 3)}') 
  
  # See a column map at middays.
  #midday = inputs[:, con.hour] == 12
  #inputs = inputs[midday]
  #targets = targets[midday]
  #preds = preds[midday]
 
  # See a column map of average performace on all J rates. 
  r2 = round(r2_score(targets, preds), 3)
  grid = fns.make_cols_map(inputs, targets, preds)
  #fns.show_diff_map('all', grid, r2)
  
  # See a column map of all H2O data.
  #targets = targets[:, 20]
  #preds = preds[:, 20]
  #r2 = round(r2_score(targets, preds), 3)
  #grid = fns.make_cols_map(inputs, targets, preds)
  #fns.show_diff_map('water', grid, r2)

  return(grid)


# Look at performance of random forest trained on surface level data.
grid1 = compare_sza_scaling('original', 'sea-level orography')
# Look at performance of random forest trained on corrected alt but surface level SZA.
grid2 = compare_sza_scaling('corrected_alt', 'corrected altitude')
# Look at performance of random forest trained on scaled SZA.
grid3 = compare_sza_scaling('corrected_alt_sza_vertical', 'corrected altitude and vertically scaled SZA')
# Look at performance of random forest trained on geometrically snd refractively scaled SZA.
grid4 = compare_sza_scaling('corrected_alt_sza_geo_refrac', 'corrected altitude and geometrically and refractively scaled SZA')

# Show a map of which model was best.
grid_same_lat = []
grid_same_lon = []
grid_best1_lat = []
grid_best1_lon = []
grid_best2_lat = []
grid_best2_lon = []
grid_best3_lat = []
grid_best3_lon = []
grid_best4_lat = []
grid_best4_lon = []
for point in range(len(grid1)):
  lat = grid1[point][0]
  lon = grid1[point][1]
  point1 = round(grid1[point][2], 3) 
  point2 = round(grid2[point][2], 3) 
  point3 = round(grid3[point][2], 3) 
  point4 = round(grid4[point][2], 3) 
  best = max([point1, point2, point3, point4])
  if best == point1 and best != point2 and best != point3 and best != point4:
    grid_best1_lat.append(lat)
    grid_best1_lon.append(lon)
  elif best == point2 and best != point1 and best != point3 and best != point4:
    grid_best2_lat.append(lat)
    grid_best2_lon.append(lon)
  elif best == point3 and best != point1 and best != point2 and best != point4:
    grid_best3_lat.append(lat)
    grid_best3_lon.append(lon)
  elif best == point4 and best != point1 and best != point2 and best != point3:
    grid_best4_lat.append(lat)
    grid_best4_lon.append(lon)
  else:
    grid_same_lat.append(lat)
    grid_same_lon.append(lon)

#plt.figure(figsize=(10,7.5))
ax = plt.axes(projection=ccrs.Mollweide()) 
plt.title(f'Columns of best performing model data for photolysis rate predictions')
plt.scatter(grid_best1_lon, grid_best1_lat, s=3, transform=ccrs.PlateCarree(), label=f'Sea-level orography and surface-level SZA ({len(grid_best1_lon)})') 
plt.scatter(grid_best2_lon, grid_best2_lat, s=3, transform=ccrs.PlateCarree(), label=f'Corrected altitude and surface-level SZA ({len(grid_best2_lon)})') 
plt.scatter(grid_best3_lon, grid_best3_lat, s=3, transform=ccrs.PlateCarree(), label=f'Corrected altitude and vertically scaled SZA ({len(grid_best3_lon)})') 
plt.scatter(grid_best4_lon, grid_best4_lat, s=3, transform=ccrs.PlateCarree(), label=f'Corrected altitude and geometrically scaled SZA with refraction ({len(grid_best4_lon)})') 
plt.scatter(grid_same_lon, grid_same_lat, s=3, transform=ccrs.PlateCarree(), label=f'No difference ({len(grid_same_lon)})') 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
ax.set_global()
ax.coastlines()
plt.show()
plt.close()
