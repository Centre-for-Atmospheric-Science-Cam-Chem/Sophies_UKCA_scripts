'''
Name: Sophie Turner.
Date: 6/9/2024.
Contact: st838@cam.ac.uk
Compile data, not on ATom flight path, from daily UM .pp file and put them in daily csv file.
Takes a long time to save.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''

import cf
import numpy as np
import pandas as pd
from math import pi
import file_paths as paths
import codes_to_names as codes

pp_file = f'{paths.UKCA_dir}/cy731a.pl20150701.pp'

# Open the UKCA file.
day = cf.read(pp_file)

field_names = ['id%UM_m01s00i266_vn1300', 'air_pressure', 'id%UM_m01s01i140_vn1300',
               'air_temperature', 'id%UM_m01s30i206_vn1300', 'id%UM_m01s50i229_vn1300',
	       'id%UM_m01s50i501_vn1300', 'id%UM_m01s50i502_vn1300', 'id%UM_m01s50i514_vn1300',
	       'id%UM_m01s50i541_vn1300', 'id%UM_m01s50i555_vn1300', 'id%UM_m01s50i559_vn1300',
	       'id%UM_m01s50i563_vn1300', 'id%UM_m01s50i567_vn1300']

col_names = ['TIME', 'ALTITUDE m', 'LATITUDE', 'LONGITUDE', 'CLOUD %', 'PRESSURE hPa', 
             'SOLAR ZENITH ANGLE', 'TEMPERATURE K', 'RELATIVE HUMIDITY', 'JNO2 NO O3P',
	     'JCH2O H HCO', 'JCH2O H2 CO', 'JMEONO2 CH3O NO2', 'JPROPANAL CH2CH3 HCO',
	     'JNO3 NO O2', 'JHOBR OH BR', 'JH2O2 OH OH', 'JO3 O2 O1D']

# How many time & space data points we want.
n = 1000

# Pick some random points in space.
times = np.random.randint(0, 24, n)
alts = np.random.randint(0, 38, n) # No altitudes over 10 km.
lats = np.random.randint(0, 144, n)
lons = np.random.randint(0, 192, n)

# Make a list for populating with data and making dataframe.
rows = []

# For every n point...
for i in range(n):
  print(f'Making point {i+1} of {n}.')
  # Make a list to hold this row.
  row = []
  # Get dimensions.
  field = day[0]
  point = field[times[i], alts[i], lats[i], lons[i]]
  time = str(point.coord('time').data)[1:20] # Up to char 19 of the date time string.
  alt = point.coord('atmosphere_hybrid_height_coordinate').array.squeeze()
  lat = point.coord('latitude').array.squeeze()
  lon = point.coord('longitude').array.squeeze()
  # Data conversions.
  alt *= 85000 # Convert to metres for 85-level, 85 km fields (DALLTH).
  # Convert UKCA longitude from 360 to +-180 format.
  if lon > 180:
    lon -= 360
  # Start the point data row with the dimensions.
  for dim in [time, alt, lat, lon]:
    row.append(dim)
    
  # For every field at this point...
  for j in range(len(field_names)):
    field_name = field_names[j]
    col_name = col_names[j+4]
    # Get field point data.
    field = day.select(field_name)[0]
    if len(field.shape) > 3:
      point = field[times[i], alts[i], lats[i], lons[i]].data.squeeze()
    else:
      # Single level fields.
      point = field[times[i], lats[i], lons[i]].data.squeeze()
    # Data conversions.
    if col_name == 'PRESSURE hPa':
      point *= 0.01 # Pascals to hPa.
    # Convert solar zenith angle from cos radians to degrees.
    elif col_name == 'SOLAR ZENITH ANGLE': 
      point = np.arccos(point) * (180.0 / pi)   
    point = np.float32(point)
    row.append(point)
  rows.append(row)

# Put it all in a dataframe.
new_data = pd.DataFrame(data=rows, columns=col_names)
new_data.set_index('TIME')
print()
print(new_data)
print()

# Save the data.
out_path = f'{paths.UKCA_dir}/random_points.csv'
print(f'Saving new dataset at {out_path}.')
new_data.to_csv(out_path)
