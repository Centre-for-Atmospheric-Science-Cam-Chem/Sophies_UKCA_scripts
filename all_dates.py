'''
Name: Sophie Turner.
Date: 3/11/2023.
Contact: st838@cam.ac.uk
Compile relevant data from daily UM .pp files and put them in daily csv files. Takes a long time to run.
It can be better to just use this script to make daily files and then
use another script to make the all in one file after, in case
this script fails somewhere and has to be restarted mid way through the data. 
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import time
import statistics
import cf
import glob
import pandas as pd
import numpy as np
import codes_to_names as codes
from math import pi

print('Reading data...')

# File paths.
dir_path = '/scratch/st838/netscratch/'
UKCA_dir = dir_path + 'nudged_J_outputs_for_ATom/'
ATom_dir = dir_path + 'ATom_MER10_Dataset/'
ATom_file = ATom_dir + 'photolysis_data.csv'

code_names = np.array(codes.code_names)


def get_date(UKCA_file):
  # Find dates of UKCA files.
  year = UKCA_file[-11:-7]
  month = UKCA_file[-7:-5]
  day = UKCA_file[-5:-3]
  return(year, month, day)


def get_ATom_day(ATom_data, UKCA_file):
  # Pick out the corresponding date from ATom.
  year, month, day = get_date(UKCA_file)
  date_str = f'{year}-{month}-{day}'
  ATom_day = ATom_data[ATom_data.index.str.contains(date_str)]
  return(ATom_day, date_str)


def get_times(ATom_day):
  # Find out what times of day we have data for.
  ATom_hours = ATom_day[ATom_day.index.str.endswith('00:00')] # DataFrame.
  # Apply a buffer so that we don't chop off start and end times which are close to an extra hour.
  if int(ATom_day.index[0][-8:-6]) < int(ATom_hours.index[0][-8:-6]): # start hours.
    if int(ATom_day.index[0][-5:-3]) <= 10: # 10 minute buffer for start minutes.
      # Add this extra time point as 1st item.
      extra = ATom_day.iloc[0:1] 
      ATom_hours = pd.concat([extra, ATom_hours])
  if int(ATom_day.index[-1][-8:-6]) != int(ATom_hours.index[-1][-8:-6]): # end hours.
    if int(ATom_day.index[-1][-5:-3]) >= 50: # 10 minute buffer for end minutes.
      # Add this as last item.  
      extra = ATom_day.iloc[-1] 
      ATom_hours = ATom_hours._append(extra)
  return(ATom_hours)
  
  
def standard_names(data):
  # Standardise the names of ATom and UM fields into 1 format.
  for name_old in data.iloc[:,4:-1].columns:
    name_new = name_old.replace('CAFS', '')
    name_new = name_new.replace('_', ' ')
    name_new = name_new.upper()  
    name_new = name_new.strip()
    data = data.rename(columns={name_old:name_new})
  return(data)
  

def match(ATom_data, UKCA_data):
  # Find which field items are present in both datasets and discard the rest.
  matches = []
  for ATom_field in ATom_data.columns:
    for UKCA_field in UKCA_data.columns:
      if ATom_field == UKCA_field:
        matches.append(ATom_field)
  ATom_data_matched = ATom_data[matches]
  UKCA_data_matched = UKCA_data[matches] 
  return(ATom_data_matched, UKCA_data_matched)  
  
  
def remove_cloudy(data1, data2):
  # Remove entries where the amount of cloud in UKCA and ATom differs by more than 10%.
  drops = []
  for timestep in data1.index:
    cloud1 = data1.loc[timestep]['CLOUD %']
    cloud2 = data2.loc[timestep]['CLOUD %']
    if abs(cloud1 - cloud2) > 10:
      drops.append(timestep)
  data1 = data1.drop(index=drops)
  data2 = data2.drop(index=drops)
  return(data1, data2)
     
  
def run_time_left(h, i, time_read, time_timestep, num_hours):
  avg_hours = statistics.mean(num_hours)
  time_day = time_read + (time_timestep * (avg_hours - i))
  time_est = time_day * (len(UKCA_files) - h) 
  remaining = round(time_est / 60)
  print(f'Estimated time remaining = {remaining} minutes')


# Open ATom dataset, already partially pre-processed.
ATom_data = pd.read_csv(ATom_file)
ATom_data = ATom_data.rename(columns={'UTC_Start_dt':'TIME', 'T':'TEMPERATURE K', 'G_LAT':'LATITUDE', 
                                      'G_LONG':'LONGITUDE', 'G_ALT':'ALTITUDE m', 'Pres':'PRESSURE hPa',
				      'cloudindicator_CAPS':'CLOUD %'})    
ATom_data = ATom_data.set_index('TIME')

# Make a DataFrame for the new hourly datasets with all dates in one.
#ATom_hourly, UKCA_hourly = pd.DataFrame(), pd.DataFrame() 

# Find dates of UKCA files.
UKCA_files = glob.glob(UKCA_dir + '/*.pp') # Just .pp files.

'''
# Debugging test.
for h in range(1):
  UKCA_file = UKCA_files[3]
  print('\nfile:', UKCA_file)
  # Open the UKCA file.
  UKCA_day = cf.read(UKCA_file)
  # Pick out the corresponding date from ATom.
  ATom_day, date = get_ATom_day(ATom_data, UKCA_file)
  if not ATom_day.empty:
    ATom_hours = get_times(ATom_day)
    timesteps = ATom_hours.index
    for i in range(len(timesteps)):    
        timestep = timesteps[i]
        # Get the UM data for the same time of day.
        hour_num = int(timestep[11:13])-1
        if hour_num > -2 and hour_num < 2:
          print('\nATom timestep:', timestep)
          print('hour index from 0:', hour_num)
          UKCA_point = UKCA_day[0][hour_num]
          print('UKCA hourly point:', UKCA_point)
exit()  
'''
'''
time_read = 279
time_timestep = 234
avg_timesteps = 5
num_hours = [avg_timesteps]
run_time_left(0, 0, time_read, time_timestep, num_hours)
'''
for h in range(len(UKCA_files)):
  start = time.time()  
  UKCA_file = UKCA_files[h]
  print('starting', UKCA_file)
  
  # Open the UKCA file.
  UKCA_day = cf.read(UKCA_file)
  
  end = time.time()
  time_read = end - start
  
  # Pick out the corresponding date from ATom.
  ATom_day, date = get_ATom_day(ATom_data, UKCA_file)
  if not ATom_day.empty:
    ATom_hours = get_times(ATom_day)
    timesteps = ATom_hours.index
    #num_hours.append(len(timesteps))

    # Table to make the matching UKCA DataFrame.
    UKCA_hours = []
    # Column names for new UKCA dataset.
    names = ['TIME', 'ALTITUDE m', 'PRESSURE hPa', 'LATITUDE', 'LONGITUDE'] 
    
    for i in range(len(timesteps)):
      start = time.time()
    
      # The lat, long, alt and pres are the same for every ATom field at each time step as it is a flight path.
      timestep = timesteps[i]
      print(timestep)
      ATom_lat = ATom_hours.loc[timestep]['LATITUDE']
      ATom_long = ATom_hours.loc[timestep]['LONGITUDE']
      ATom_alt = ATom_hours.loc[timestep]['ALTITUDE m']
      ATom_pres = ATom_hours.loc[timestep]['PRESSURE hPa']  
      
      # Get the UM data for the same time of day.
      hour_num = int(timestep[11:13])-1
      UKCA_point = UKCA_day[0][hour_num]
      
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
      
      # Convert UKCA longitude from 360 to 180 format.
      UKCA_point_long = float(np.squeeze(UKCA_longs[idx_long]))
      if UKCA_point_long > 180:
        UKCA_point_long -= 360
      
      # The Nones are placeholders for altitude and pressure.
      UKCA_point_entry = [timestep, None, None, float(np.squeeze(UKCA_lats[idx_lat])), UKCA_point_long]
 
      # Match each item by hourly time steps.
      for j in range(len(UKCA_day)):  
        UKCA_point = UKCA_day[j][i+9] # Get the UM data for the same time of day.
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
              UKCA_point_entry[2] = float(np.squeeze(UKCA_pressures[idx_pres]))
            value = UKCA_point[idx_pres,idx_lat,idx_long].data  
          else:
            # Height levels.  
            UKCA_alts = UKCA_point.coord('atmosphere_hybrid_height_coordinate').data
            UKCA_alts = UKCA_alts * 85000 # Convert to metres for 85-level, 85 km fields (DALLTH).
            diffs = np.absolute(UKCA_alts - ATom_alt)
            idx_alt = diffs.argmin()
	    # Add the altitude.
            UKCA_point_entry[1] = float(np.squeeze(UKCA_alts[idx_alt]))
            value = UKCA_point[idx_alt,idx_lat,idx_long].data
        # Some fields are 2D with only 1 vertical level.
        elif UKCA_point.ndim == 2:
          value = UKCA_point[idx_lat,idx_long].data 
        # Add the value to the list as a number.
        value = float(np.squeeze(value))
        # Add the J rate or other field value.
        if name == 'PRESSURE AT THETA LEVELS AFTER TS':
          UKCA_point_entry[2] = value * 0.01 # Pascals to hPa.
        else: 
          # Convert solar zenith angle from cos radians to degrees.
          if name == 'COS SOLAR ZENITH ANGLE':
            value = np.arccos(value) * 180 / pi
            name = 'SOLAR ZENITH ANGLE'
          UKCA_point_entry.append(value)
          if name not in names:
            names.append(name) 
      # Add a row of data to the table for this timestep.
      UKCA_hours.append(UKCA_point_entry)
      '''
      end = time.time()
      time_timestep = end - start 
      run_time_left(h, i, time_read, time_timestep, num_hours)
      '''
    # Turn the selected UKCA data into a DataFrame to match the ATom one and standardise both.  
    UKCA_hours = pd.DataFrame(data=(UKCA_hours), columns=names)
    UKCA_hours = UKCA_hours.set_index('TIME')
    UKCA_hours = standard_names(UKCA_hours)
    UKCA_hours = UKCA_hours.rename(columns={'TEMPERATURE ON THETA LEVELS':'TEMPERATURE K', 
                              'BULK CLOUD FRACTION IN EACH LAYER':'CLOUD %',
			      'RELATIVE HUMIDITY ON P LEV/UV GRID':'RELATIVE HUMIDITY'})
    ATom_hours = standard_names(ATom_hours)  
    
    # Only use the match function if you want to remove columns that can't be directly compared.
    # To retain all the fields, comment out this line.
    ATom_hours, UKCA_hours = match(ATom_hours, UKCA_hours)
    
    # Remove datapoints with differing clouds. Comment out this line to keep all cloudy points.
    ATom_hours, UKCA_hours = remove_cloudy(ATom_hours, UKCA_hours)
    
    # Save daily hourly data for each day separately.
    ATom_out_path = f'{ATom_dir}/ATom_hourly_{date}.csv'
    # Sort by date, not by file name.
    ATom_hours = ATom_hours.sort_index()
    ATom_hours.to_csv(ATom_out_path) 
    UKCA_out_path = f'{UKCA_dir}/UKCA_hourly_{date}.csv'
    # Sort by date, not by file name.
    UKCA_hours = UKCA_hours.sort_index()
    UKCA_hours.to_csv(UKCA_out_path)
    '''
    # Build up the big datasets of all the hourly points.
    ATom_hourly = ATom_hourly._append(ATom_hours) 
    UKCA_hourly = UKCA_hourly._append(UKCA_hours)
    '''
'''    
# Save the hourly datasets for all days.
ATom_out_path = f'{ATom_dir}/ATom_hourly_all.csv'
ATom_hourly.to_csv(ATom_out_path)
UKCA_out_path = f'{UKCA_dir}/UKCA_hourly_all.csv'
UKCA_hourly.to_csv(UKCA_out_path)
'''    
    

    

 
