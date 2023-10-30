'''
Name: Sophie Turner.
Date: 10/10/2023.
Contact: st838@cam.ac.uk
Pick out relevant data from ATom and UM outputs and put them into simple csv files.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

# Stop unnecessary warnings about pandas version.
import warnings
warnings.simplefilter('ignore')

import cf
import pandas as pd 
import numpy as np
import codes_to_names as codes

date = '2017-10-23'

# File paths.
dir_path = '/scratch/st838/netscratch/'
UKCA_dir = 'StratTrop_nudged_J_outputs_for_ATom/'
ATom_dir = 'ATom_MER10_Dataset.20210613/'
UKCA_file = 'tests/cy731a.pl20150129.pp' # A test file with extra physical fields. Not for actual use with ATom.
ATom_file = dir_path + ATom_dir + 'photolysis_data.csv'
UKCA_file = dir_path + UKCA_file

code_names = np.array(codes.code_names)


def standard_names(data):
  # Standardise the names of ATom and UM fields into 1 format.
  for name_old in data.iloc[:,4:-1].columns:
    name_new = name_old.replace('CAFS', '')
    name_new = name_new.replace('_', ' ')
    name_new = name_new.upper()  
    name_new = name_new.strip()
    data = data.rename(columns={name_old:name_new})
  return(data)
  

# Open the .csv of all the ATom data which I have already pre-processed in the script, ATom_J_data.
print('Loading the ATom data.')
ATom_data = pd.read_csv(ATom_file) # dims = 2. time steps, chemicals + spatial dimensions.

# This step takes several minutes. Better to load individual chunks than the whole thing. See stash codes text file.
print('Loading the UM data.')
UKCA_data = cf.read(UKCA_file)

print('Refining by time comparison.')

ATom_data = ATom_data.rename(columns={'UTC_Start_dt':'TIME', 'T':'TEMPERATURE K', 'G_LAT':'LATITUDE', 
                                      'G_LONG':'LONGITUDE', 'G_ALT':'ALTITUDE m', 'Pres':'PRESSURE hPa'})    
ATom_data = ATom_data.set_index('TIME')

# Pick out the fields.
ATom_data = ATom_data[ATom_data.index.str.contains(date)]

# ATom and UM are using UTC.
timesteps = ['2017-10-23 10:00:00', '2017-10-23 11:00:00', '2017-10-23 12:00:00', '2017-10-23 13:00:00', '2017-10-23 14:00:00', 
             '2017-10-23 15:00:00', '2017-10-23 16:00:00', '2017-10-23 17:00:00', '2017-10-23 18:00:00', '2017-10-23 18:59:00']
# Trim so that data are over the same times.
ATom_data = ATom_data.loc[timesteps] # dims = 2. type = pandas dataframe. 

names = ['TIME', 'ALTITUDE m', 'PRESSURE hPa', 'LATITUDE', 'LONGITUDE']  
table_data = []

for i in range(len(timesteps)):
  # The lat, long, alt and pres are the same for every ATom field at each time step as it is a flight path.
  time = timesteps[i]
  ATom_lat = ATom_data.loc[time]['LATITUDE']
  ATom_long = ATom_data.loc[time]['LONGITUDE']
  ATom_alt = ATom_data.loc[time]['ALTITUDE m']
  ATom_pres = ATom_data.loc[time]['PRESSURE hPa']
  
  # Pick a point from UKCA for these times.
  UKCA_point = UKCA_data[0][i+9] # Get the UM data for the same time of day.
  
  # Match latitude.
  UKCA_lats = UKCA_point.coord('latitude').data
  diffs = np.absolute(UKCA_lats - ATom_lat)
  idx_lat = diffs.argmin()
  
  # Match longitude. ATom: degrees +E in half circles (-180 to 180). UM: degrees in a full circle (0 to 360).
  UKCA_longs = UKCA_point.coord('longitude').data
  if ATom_long < 0:
    ATom_long = ATom_long + 360
  diffs = np.absolute(UKCA_longs - ATom_long)
  idx_long = diffs.argmin()
  
  # The Nones are placeholders for altitude and pressure.
  UKCA_point_entry = [time, None, None, np.squeeze(UKCA_lats[idx_lat]), np.squeeze(UKCA_longs[idx_long])]
 
  # Match each item by hourly time steps.
  for j in range(len(UKCA_data)):  
    UKCA_point = UKCA_data[j][i+9] # Get the UM data for the same time of day.
    name = UKCA_point.long_name    
    
    # Convert name if needed.
    if name in code_names[:,0]:
      i_name = np.where(code_names[:,0] == name)
      name = code_names[i_name,1][0,0] 
      UKCA_point.long_name = name
      
    UKCA_point = np.squeeze(UKCA_point) # Get dimensions.
    # Some fields are 3D with differing vertical levels.
    if UKCA_point.ndim == 3:
      if UKCA_point.shape[0] < 85:
        # Pressure levels.
        UKCA_pressures = UKCA_point.coord('air_pressure').data
        diffs = np.absolute(UKCA_pressures - ATom_pres)
        idx_pres = diffs.argmin()
	# Add this pressure if there is no better value for pressure.
        if UKCA_point_entry[2] is None:
          UKCA_point_entry[2] = np.squeeze(UKCA_pressures[idx_pres])
        value = UKCA_point[idx_pres,idx_lat,idx_long].data  
      else:
        # Height levels.  
        UKCA_alts = UKCA_point.coord('atmosphere_hybrid_height_coordinate').data
        UKCA_alts = UKCA_alts * 85000 # Convert to metres for 85-level, 85 km fields (DALLTH).
        diffs = np.absolute(UKCA_alts - ATom_alt)
        idx_alt = diffs.argmin()
	# Add the altitude.
        UKCA_point_entry[1] = np.squeeze(UKCA_alts[idx_alt])
        value = UKCA_point[idx_alt,idx_lat,idx_long].data
    # Some fields are 2D with only 1 vertical level.
    elif UKCA_point.ndim == 2:
      value = UKCA_point[idx_lat,idx_long].data 
    value = np.squeeze(value)
    # Add the J rate or other field value.
    if name == 'PRESSURE AT THETA LEVELS AFTER TS':
      UKCA_point_entry[2] = value * 0.01 # Pascals to hPa.
    else: 
      # More conversions.
      if name == 'COS SOLAR ZENITH ANGLE':
        value = np.arccos(value)
        name = 'SOLAR ZENITH ANGLE'
      # Add the value to the list as a number.
      value = float(value)
      UKCA_point_entry.append(value)
      if name not in names:
        names.append(name) 
  # Add a row of data to the table for this timestep.
  table_data.append(UKCA_point_entry)
  
table = pd.DataFrame(data=(table_data), columns=names)
table = table.set_index('TIME')
table = standard_names(table)
table = table.rename(columns={'TEMPERATURE ON THETA LEVELS':'TEMPERATURE K'})
ATom_data = standard_names(ATom_data)

# Make a csv
ATom_out_path = f'{dir_path}tests/ATom_points_{date}.csv'
ATom_data.to_csv(ATom_out_path)
UKCA_out_path = f'{dir_path}tests/UKCA_points_{date}.csv'
table.to_csv(UKCA_out_path)
