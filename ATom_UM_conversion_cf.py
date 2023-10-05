'''
Name: Sophie Turner.
Date: 28/9/2023.
Contact: st838@cam.ac.uk
Script to make ATom data comparable with UKCA data, using cfPython.
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


# Converting with different vertical levels.
def vertical(ATom_data, UKCA_data, z_axis, field, timesteps):
  # z-axis is 'G_ALT', 'Pres' or None.
  # field is ATom's name for the thing we're comparing. 
  UKCA_points, ATom_points = [], []
  print('\nMatching points from the datasets.')

  if z_axis is not None:  
    ATom_data = ATom_data[['G_LAT', 'G_LONG', z_axis, field]] 

    for i in range(len(timesteps)):

      UKCA_point = UKCA_data[i]

      # Check what units of altitude they're using. ATom: m.
      # UM values from 0.00022 to 0.96.
      # Find out what the altitudes are for ATom and select the most similar altitudes of UKCA data.
      ATom_z = ATom_data.loc[timesteps[i]][z_axis]
      if z_axis == 'G_ALT':
        UKCA_z = UKCA_point.coord('atmosphere_hybrid_height_coordinate').data
        UKCA_z = UKCA_z * 85000 # Convert to metres for 85-level, 85 km fields (DALLTH).
      elif z_axis == 'Pres':
        UKCA_z = UKCA_point.coord('air_pressure').data
      diffs = np.absolute(UKCA_z - ATom_z)
      iZ = diffs.argmin()
      UKCA_point = UKCA_point[0,iZ]

      # Check what units of latitude they're using. ATom: degrees +N. UM: degrees.
      # Find out what the latitudes are for ATom and select the most similar latitudes of UKCA data.
      ATom_lat = ATom_data.loc[timesteps[i]]['G_LAT']
      UKCA_lats = UKCA_point.coord('latitude').data
      diffs = np.absolute(UKCA_lats - ATom_lat)
      iLat = diffs.argmin()
      UKCA_point = UKCA_point[0,0,iLat]

      # Check what units of longitude they're using. ATom: degrees +E in half circles (-180 to 180). UM: degrees in a full circle (0 to 360).
      # Find out what the longtitudes are for ATom and select the most similar longitudes of UKCA data.
      ATom_long = ATom_data.loc[timesteps[i]]['G_LONG']
      UKCA_longs = UKCA_point.coord('longitude').data
      if ATom_long < 0:
        ATom_long = ATom_long + 360
      diffs = np.absolute(UKCA_longs - ATom_long)
      iLong = diffs.argmin()
      UKCA_point = UKCA_point[0,0,0,iLong]
  
      UKCA_points.append(UKCA_point) # list of cf fields.
      ATom_points.append(ATom_data.loc[timesteps[i]]) # list of pandas.core.series.Series.
  
  else:
    ATom_data = ATom_data[['G_LAT', 'G_LONG', field]] 

    for i in range(len(timesteps)):

      UKCA_point = UKCA_data[i]

      # Check what units of latitude they're using. ATom: degrees +N. UM: degrees.
      # Find out what the latitudes are for ATom and select the most similar latitudes of UKCA data.
      ATom_lat = ATom_data.loc[timesteps[i]]['G_LAT']
      UKCA_lats = UKCA_point.coord('latitude').data
      diffs = np.absolute(UKCA_lats - ATom_lat)
      iLat = diffs.argmin()
      UKCA_point = UKCA_point[0,iLat]

      # Check what units of longitude they're using. ATom: degrees +E in half circles (-180 to 180). UM: degrees in a full circle (0 to 360).
      # Find out what the longtitudes are for ATom and select the most similar longitudes of UKCA data.
      ATom_long = ATom_data.loc[timesteps[i]]['G_LONG']
      UKCA_longs = UKCA_point.coord('longitude').data
      if ATom_long < 0:
        ATom_long = ATom_long + 360
      diffs = np.absolute(UKCA_longs - ATom_long)
      iLong = diffs.argmin()
      UKCA_point = UKCA_point[0,0,iLong]
  
      UKCA_points.append(UKCA_point) # list of cf fields.
      ATom_points.append(ATom_data.loc[timesteps[i]]) # list of pandas.core.series.Series.
  
  # Checking that they match.
  ATom_point = ATom_data.loc[timesteps[i]]
  print('\nATom last data point:')
  print(type(ATom_point)) # pandas.core.series.Series
  print(ATom_point.shape) # (4,)
  print(ATom_point)
  print('\nUM data point closest spatial match:')
  print(type(UKCA_point)) # cf.field.
  if z_axis == 'G_ALT':
    print('Altitude:', UKCA_point.coord('atmosphere_hybrid_height_coordinate').data * 85000)
  print(UKCA_point.shape) # (1,1,1,1)
  print(UKCA_point)
  print(UKCA_point.data)
    
  return(ATom_data, ATom_points, UKCA_data, UKCA_points)


dates = ['2016-08-06', '2017-10-23', '2018-05-12']

# File paths.
UKCA_dir = './StratTrop_nudged_J_outputs_for_ATom/'
ATom_dir = './ATom_MER10_Dataset.20210613/'
#UKCA_file = '{}cy731a.pl{}.pp'.format(UKCA_dir, dates[1].replace('-', ''))
UKCA_file = './tests/cy731a.pl20150129.pp' # A test file with extra physical fields. Not for actual use with ATom.
ATom_file = ATom_dir + 'photolysis_data.csv'

# Stash codes.
UM_JO3 = 'stash_code=50567' # O3 -> O(1D) + O2. 24,85,144,192.
UM_JNO2 = 'stash_code=50574' # NO2 -> NO + O. 24,85,144,192.
UM_cloud = 'stash_code=266' # Cloud fraction, 0-1 volume of grid box. 24,85,144,192. 
UM_zenith = 'stash_code=1142' # Cosine of solar zenith angle. 24,144,192.
UM_humid_s = 'stash_code=10' # Specific humidity, kg kg-1. 24,85,144,192.
UM_humid_r = 'stash_code=30206' # Relative humidity on P levels. 24,36,144,192. 
UM_temp = 'stash_code=16004' # Temperature, K. 24,85,144,192.
UM_pressure = 'stash_code=408' # Pressure, Pa. 24,85,144,192.

# ATom field names.
ATom_JO3 = 'jO3_O2_O1D_CAFS' # O3 -> O(1D) + O2
ATom_JNO2 = 'jNO2_NO_O3P_CAFS' # NO2 -> NO + O
ATom_cloud = 'CloudFlag_AMS' # Cloud fraction?
ATom_zenith = 'Solar_Zenith_Angle' # Solar zenith angle, deg. 
ATom_humid = 'Relative_Humidity' # Relative humidity, %.
ATom_temp = 'T' # Temperature, K. 
ATom_pressure = 'Pres' # Pressure, hPa.

# What we want to compare.
UM_field = UM_zenith
ATom_field = ATom_zenith
print('Comparing', ATom_field)

# This step takes several minutes. Better to load individual chunks than the whole thing. See stash codes text file.
print('\nLoading the UM data.')
UKCA_data = cf.read(UKCA_file,select=UM_field)[0] 
'''
# Investigate the contents.
print(type(UKCA_data)) # cf field
print(UKCA_data.shape) # 24 t, 85 z, 144 y, 192 x
print(UKCA_data) # an overview of cf field
print(UKCA_data.data) # [[[[3.63421518898531e-14, ..., 0.0]]]]
'''
# Get the vertical levels.
try:
  UKCA_data.coord('atmosphere_hybrid_height_coordinate')
  z_axis = 'G_ALT'
except:
  try:
    UKCA_data.coord('air_pressure')
    z_axis = 'Pres'
  except:
    z_axis = None

# Open the .csv of all the ATom data which I have already pre-processed in the script, ATom_J_data.
print('\nLoading the ATom data.')
ATom_data = pd.read_csv(ATom_file, index_col=0) # dims = 2. time steps, chemicals + spatial dimensions.

# Pick out the fields.
ATom_data = ATom_data[ATom_data.index.str.contains(dates[1])]

# Check what the time of day range is.
#print(ATom_data.index[0]) # 09:33
#print(ATom_data.index[-1], '\n') # 18:59
# Trim so that all arrays are over the same times. # 09:33:00 to 18:59:00
ATom_data = ATom_data.loc['2017-10-23 09:33:00':'2017-10-23 18:59:00'] # dims= 2. type = pandas dataframe.

# Make arrays for a direct comparison of hourly time steps.
# ATom and UM are using UTC.
# Get the UM data for the same time of day.
UKCA_data = UKCA_data[9:19] # 10:00 to 19:00. dims = 4. t, z, y, x. type = cf field.

timesteps = ['2017-10-23 10:00:00', '2017-10-23 11:00:00', '2017-10-23 12:00:00', '2017-10-23 13:00:00', '2017-10-23 14:00:00', 
             '2017-10-23 15:00:00', '2017-10-23 16:00:00', '2017-10-23 17:00:00', '2017-10-23 18:00:00', '2017-10-23 18:59:00']

ATom_data, ATom_points, UKCA_data, UKCA_points = vertical(ATom_data, UKCA_data, z_axis, ATom_field, timesteps)

# Conversion for cos solar zenith angle.
if UM_field == UM_zenith:
  UKCA_points = np.arccos(UKCA_points)
  print(UKCA_points[-1])
  
# For this example, ATom J rate measurements are around e-5 but UKCA J rates are around e-11. 
# Find out if this is common in the data. 
# See if I can tell UM to replicate clouds from ATom. 
# Check the sources ATom and Fast-J use for x-sections and quantum yields and see if they cause this. 
# compare ATom days with each other for O3.
# compare UM days with each other for O3.
# compare the ATom days with the UM days.
# put all the above into functions to work with different chemicals and times etc and try with NO2.
# compare ratios of other rates to NO2 at same points for both UCKA and ATom.

