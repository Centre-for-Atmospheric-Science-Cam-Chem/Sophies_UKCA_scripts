'''
Name: Sophie Turner.
Date: 28/9/2023.
Contact: st838@cam.ac.uk
Script to make ATom data comparable with UKCA data, using iris. 
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
Occasionally, the run terminates unexpectedly after giving a warning but no error. Possibly due to remote connection. Run it again and it should work.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

# Stop annoying warnings about pandas version.
import warnings
warnings.simplefilter('ignore')

import iris 
import numpy as np 
import pandas as pd 

dates = ['2016-08-06', '2017-10-23', '2018-05-12']

# File paths.
UKCA_dir = './StratTrop_nudged_J_outputs_for_ATom/'
ATom_dir = './ATom_MER10_Dataset.20210613/'
UKCA_file = '{}cy731a.pl{}.pp'.format(UKCA_dir, dates[1].replace('-', ''))
#UKCA_file = UKCA_dir + '20150101test.pp' # A test file not for actual use with ATom.
ATom_file = ATom_dir + 'photolysis_data.csv'

# Stash codes.
UM_JO3 = 'm01s50i567' # O3 -> O(1D) + O2
UM_JNO2 = 'm01s50i574' # NO2 -> NO + O
UM_cloud = 'm01s00i266' # Cloud fraction, 0-1 volume of grid box. cube.data gives an unpack error.
UM_sw_down = 'm01s01i218' # Downward shortwave radiation flux, rho grid, Wm-2.
UM_sw_up = 'm01s01i217' # Upward shortwave radiation flux, rho grid, Wm-2.
UM_zenith = 'm01s01i142' # Cosine of solar zenith angle. error: found no level heights.
UM_humid = 'm01s00i010' # Specific humidity, kg kg-1.
UM_temp = 'm01s16i004' # Temperature, K. error: found no level heights.
UM_pressure = 'm01s00i408' # Pressure, Pa. cube.data gives an unpack error.

# ATom field names.
ATom_JO3 = 'jO3_O2_O1D_CAFS' # O3 -> O(1D) + O2
ATom_JNO2 = 'jNO2_NO_O3P_CAFS' # NO2 -> NO + O
ATom_cloud = 'CloudFlag_AMS' # Cloud fraction?
ATom_down_O3 = 'jO3_dwnFrac_CAFS' # Total radiation flux in O3.
ATom_zenith = 'Solar_Zenith_Angle' # Solar zenith angle, deg.
ATom_humid = 'Relative_Humidity' # Relative humidity, %.
ATom_temp = 'T' # Temperature, K. 
ATom_pressure = 'Pres' # Pressure, hPa.

# This step takes several minutes. Better to load individual chunks than the whole thing. See stash codes text file.
print('\nLoading the UM data.')
UKCA_data = iris.load_cube(UKCA_file,iris.AttributeConstraint(STASH=UM_JO3)) 

# Investigate the contents.
print(type(UKCA_data)) # iris cube
print(UKCA_data.shape) # 24 t, 85 z, 144 y, 192 x
print(UKCA_data)
print(UKCA_data.data)
'''
# Open the .csv of all the ATom data which I have already pre-processed in the script, ATom_J_data.
print('\nLoading the ATom data.')
ATom_data = pd.read_csv(ATom_file, index_col=0) # dims = 2. time steps, chemicals + spatial dimensions.

# Pick out the fields.
ATom_data = ATom_data[ATom_data.index.str.contains(dates[1])]
ATom_data = ATom_data[['G_LAT', 'G_LONG', 'G_ALT', ATom_JO3]] 

# Check what the time of day range is.
#print(ATom_data.index[0]) # 09:33
#print(ATom_data.index[-1], '\n') # 18:59

# Trim so that all arrays are over the same times. # 09:33:00 to 18:59:00
ATom_data = ATom_data.loc['2017-10-23 09:33:00':'2017-10-23 18:59:00'] # dims= 2. type = pandas dataframe.

# Make arrays for a direct comparison of hourly time steps.
# ATom and UM are using UTC.
# Get the UM data for the same time of day.
UKCA_data = UKCA_data[8:18] # 10:00 to 19:00. dims = 4. t, z, y, x. type = iris cube.

timesteps = ['2017-10-23 10:00:00', '2017-10-23 11:00:00', '2017-10-23 12:00:00', '2017-10-23 13:00:00', '2017-10-23 14:00:00', 
             '2017-10-23 15:00:00', '2017-10-23 16:00:00', '2017-10-23 17:00:00', '2017-10-23 18:00:00', '2017-10-23 18:59:00']
UKCA_points, ATom_points = [], []

print('\nMatching points from the datasets.')
for i in range(len(timesteps)):

  UKCA_point = UKCA_data[i]

  # Check what units of altitude they're using. ATom: m. UM: m.
  # Find out what the altitudes are for ATom and select the most similar altitudes of UKCA data.
  ATom_alt = ATom_data.loc[timesteps[i]]['G_ALT']
  UKCA_alts = UKCA_point.coord('level_height').points
  diffs = np.absolute(UKCA_alts - ATom_alt)
  iAlt = diffs.argmin()
  UKCA_point = UKCA_point[iAlt]

  # Check what units of latitude they're using. ATom: degrees +N. UM: degrees.
  # Find out what the latitudes are for ATom and select the most similar latitudes of UKCA data.
  ATom_lat = ATom_data.loc[timesteps[i]]['G_LAT']
  UKCA_lats = UKCA_point.coord('latitude').points
  diffs = np.absolute(UKCA_lats - ATom_lat)
  iLat = diffs.argmin()
  UKCA_point = UKCA_point[iLat]

  # Check what units of longitude they're using. ATom: degrees +E in half circles (-180 to 180). UM: degrees in a full circle (0 to 360).
  # Find out what the longtitudes are for ATom and select the most similar longitudes of UKCA data.
  ATom_long = ATom_data.loc[timesteps[i]]['G_LONG']
  UKCA_longs = UKCA_point.coord('longitude').points
  if ATom_long < 0:
    ATom_long = ATom_long + 360
  diffs = np.absolute(UKCA_longs - ATom_long)
  iLong = diffs.argmin()
  UKCA_point = UKCA_point[iLong]
  
  UKCA_points.append(UKCA_point) # list of iris.cube.Cube.
  ATom_points.append(ATom_data.loc[timesteps[i]]) # list of pandas.core.series.Series.

# Checking that they match.  
ATom_point = ATom_data.loc[timesteps[i]]
print('\nATom last data point:')
print(type(ATom_point)) # pandas.core.series.Series
print(ATom_point.shape) # (4,)
print(ATom_point)
print('\nUM data point closest spatial match:')
print(type(UKCA_point)) # iris.cube.Cube
print(UKCA_point.shape) # ()
print(UKCA_point)
print(UKCA_point.data)

# For this example, ATom J rate measurements are around e-5 but UKCA J rates are around e-11. 
# Find out if this is common in the data. 
# See if I can tell UM to replicate clouds from ATom. 
# Check the sources ATom and Fast-J use for x-sections and quantum yields and see if they cause this. 
# compare ATom days with each other for O3.
# compare UM days with each other for O3.
# compare the ATom days with the UM days.
# put all the above into functions to work with different chemicals and times etc and try with NO2.
# compare ratios of other rates to NO2 at same points for both UCKA and ATom.

#outpath = './test_cube.nc'
#iris.save(UKCA_data, outpath)
'''
