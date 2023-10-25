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

date = '2017-10-23'

# File paths.
dir_path = '/scratch/st838/netscratch/'
UKCA_dir = 'StratTrop_nudged_J_outputs_for_ATom/'
ATom_dir = 'ATom_MER10_Dataset.20210613/'
UKCA_file = 'tests/cy731a.pl20150129.pp' # A test file with extra physical fields. Not for actual use with ATom.
ATom_file = dir_path + ATom_dir + 'photolysis_data.csv'
UKCA_file = dir_path + UKCA_file

# Open the .csv of all the ATom data which I have already pre-processed in the script, ATom_J_data.
print('Loading the ATom data.')
ATom_data = pd.read_csv(ATom_file, index_col=0) # dims = 2. time steps, chemicals + spatial dimensions.

# This step takes several minutes. Better to load individual chunks than the whole thing. See stash codes text file.
print('Loading the UM data.')
UKCA_data = cf.read(UKCA_file)

print('\nRefining by time comparison.')

# Pick out the fields.
ATom_data = ATom_data[ATom_data.index.str.contains(date)]

# ATom and UM are using UTC.
timesteps = ['2017-10-23 10:00:00', '2017-10-23 11:00:00', '2017-10-23 12:00:00', '2017-10-23 13:00:00', '2017-10-23 14:00:00', 
             '2017-10-23 15:00:00', '2017-10-23 16:00:00', '2017-10-23 17:00:00', '2017-10-23 18:00:00', '2017-10-23 18:59:00']
# Trim so that data are over the same times.
ATom_data = ATom_data.loc[timesteps] # dims = 2. type = pandas dataframe.       

for i in range(len(timesteps)):
  # The lat, long, alt and pres are the same for every ATom field at each time step as it is a flight path.
  time = timesteps[i]
  ATom_lat = ATom_data.loc[time]['G_LAT']
  ATom_long = ATom_data.loc[time]['G_LONG']
  ATom_alt = ATom_data.loc[time]['G_ALT']
  ATom_pres = ATom_data.loc[time]['Pres']
  
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
  UKCA_point_entry = [time, None, None, UKCA_lats[idx_lat], UKCA_longs[idx_long]]
 
  # Match each item by hourly time steps.
  for j in range(len(UKCA_data)):
    UKCA_point = UKCA_data[j][i+9] # Get the UM data for the same time of day.
    UKCA_point = np.squeeze(UKCA_point) # Get dimensions.
    # Some fields are 3D with differing vertical levels.
    if UKCA_point.ndim == 3:
      if UKCA_point.shape[0] < 85:
        # Pressure levels.
        UKCA_pressures = UKCA_point.coord('air_pressure').data
        diffs = np.absolute(UKCA_pressures - ATom_pres)
        idx_pres = diffs.argmin()
	# Add pressure.
        UKCA_point_entry[2] = UKCA_pressures[idx_pres] 
	# Add the J rate or other field value.
        UKCA_point_entry.append(UKCA_point[idx_pres,idx_lat,idx_long].data)
      else:
        # Height levels.  
        UKCA_alts = UKCA_point.coord('atmosphere_hybrid_height_coordinate').data
        UKCA_alts = UKCA_alts * 85000 # Convert to metres for 85-level, 85 km fields (DALLTH).
        diffs = np.absolute(UKCA_alts - ATom_alt)
        idx_alt = diffs.argmin()
	# Add the altitude.
        UKCA_point_entry[1] = UKCA_alts[idx_alt]
        # Add the J rate or other field value.
        UKCA_point_entry.append(UKCA_point[idx_alt,idx_lat,idx_long].data)
    # Some fields are 2D with only 1 vertical level.
    elif UKCA_point.ndim == 2:
      # Add the J rate or other field value.
      UKCA_point_entry.append(UKCA_point[idx_lat,idx_long].data)
    
  print('ATom altitude:')
  print(ATom_data.loc[time]['G_ALT'])
  print('UKCA altitude:')
  print(UKCA_point_entry[1])
  print('ATom pressure:')
  print(ATom_data.loc[time]['Pres'])
  print('UKCA pressure from PLEV:')
  print(UKCA_point_entry[2])
  print('UKCA pressure from DALLTH:')  
  print(UKCA_point_entry[7])
  # Note how the PLEV is not as accurate as the DALLTH.
  # Might need more PLEV vertical increments. We only have 36.
  # What is PLEV for? Do we need it to be perfect? If not, replace [2] with DALLTH pressure output.
  exit()    
    
     
          
'''
# Make a .csv.
ATom_out_path = f'./tests/ATom_points_{date}.csv'
ATom_data.to_csv(ATom_out_path)
UKCA_out_path = f'./tests/UKCA_points_{date}.csv'

'''
