"""
Scripts to make ATom data comparable with UKCA data. 
Occasionally, the run terminates unexpectedly after giving a warning but no error. Possibly due to remote connection. Run it again and it should work.
"""
# Necessary pip commands are in iris_imports.sh. 

import iris # pip install scitools-iris
import matplotlib.pyplot as plt
import numpy as np # pip install numpy==1.21
import pandas as pd # version 2.0.3. pip install --upgrade pandas if needed.

# File paths.
UKCA_dir = './StratTrop_nudged_J_outputs_for_ATom/'
ATom_dir = './ATom_MER10_Dataset.20210613/'
UKCA_file = UKCA_dir + 'cy731a.pl20171023.pp'
ATom_file = ATom_dir + 'photolysis_data.csv'

# This step takes several minutes. Better to load individual chunks than the whole thing. See stash codes text file.
UKCA_O3 = iris.load_cube(UKCA_file,iris.AttributeConstraint(STASH='m01s50i567')) # O3 --> O(1D)

# Investigate the contents.
'''
print(type(UKCA_O3)) # iris cube
print(UKCA_O3.shape) # 24 t, 85 z, 144 y, 192 x
print(UKCA_O3)
'''

# Both the files are for 23rd October 2017.

# Open the .csv of all the ATom data which I have already pre-processed in the script, ATom_J_data.
ATom_data = pd.read_csv(ATom_file, index_col=0) # dims = 2. time steps, chemicals + spatial dimensions.

# Pick out the fields.
ATom_day = ATom_data[ATom_data.index.str.contains('2017-10-23')] 
ATom_O3 = ATom_day[['G_LAT', 'G_LONG', 'G_ALT', 'jO3_O2_O1D_CAFS']] 

# Check what the time of day range is.
#print(ATom_O3.index[0]) # 09:33
#print(ATom_O3.index[-1], '\n') # 18:59

# Trim so that all arrays are over the same times. # 09:33:00 to 18:59:00
ATom_O3 = ATom_O3.loc['2017-10-23 09:33:00':'2017-10-23 18:59:00'] # dims= 2. type = pandas dataframe.

# Make arrays for a direct comparison of hourly time steps.
# ATom and UM are using UTC.
# Get the UM data for the same time of day.
UKCA_O3 = UKCA_O3[9:19] # 10:00 to 19:00. dims = 4. t, z, y, x. type = iris cube.

timesteps = ['2017-10-23 10:00:00', '2017-10-23 11:00:00', '2017-10-23 12:00:00', '2017-10-23 13:00:00', '2017-10-23 14:00:00', 
             '2017-10-23 15:00:00', '2017-10-23 16:00:00', '2017-10-23 17:00:00', '2017-10-23 18:00:00', '2017-10-23 18:59:00']
UKCA_points, ATom_points = [], []

for i in range(len(timesteps)):

  UKCA_point = UKCA_O3[i]

  # Check what units of altitude they're using. ATom: m. UM: m.
  # Find out what the altitudes are for ATom and select the right altitudes of UKCA data.
  ATom_alt = ATom_O3.loc[timesteps[i]]['G_ALT']
  UKCA_alts = UKCA_point.coord('level_height').points
  diffs = np.absolute(UKCA_alts - ATom_alt)
  iAlt = diffs.argmin()
  UKCA_point = UKCA_point[iAlt]

  # Check what units of latitude they're using. ATom: degrees +N. UM: degrees.
  # Find out what the latitudes are for ATom and select the right latitudes of UKCA data.
  ATom_lat = ATom_O3.loc[timesteps[i]]['G_LAT']
  UKCA_lats = UKCA_point.coord('latitude').points
  diffs = np.absolute(UKCA_lats - ATom_lat)
  iLat = diffs.argmin()
  UKCA_point = UKCA_point[iLat]

  # Check what units of longitude they're using. ATom: degrees +E in half circles (-180 to 180). UM: degrees in a full circle (0 to 360).
  # Find out what the longtitudes are for ATom and select the right longitudes of UKCA data.
  ATom_long = ATom_O3.loc[timesteps[i]]['G_LONG']
  UKCA_longs = UKCA_point.coord('longitude').points
  if ATom_long < 0:
    ATom_long = ATom_long + 360
  diffs = np.absolute(UKCA_longs - ATom_long)
  iLong = diffs.argmin()
  UKCA_point = UKCA_point[iLong]

  # Checking that they match. They do.
  '''
  ATom_point = ATom_O3.loc[timesteps[i]]
  print('\n\n')
  print(type(UKCA_point)) # iris.cube.Cube
  print(UKCA_point.shape) # ()
  print(UKCA_point)
  print(UKCA_point.data)
  print('\n\n')
  print(type(ATom_point)) # pandas.core.series.Series
  print(ATom_point.shape) # (4,)
  print(ATom_point)
  '''
  
  UKCA_points.append(UKCA_point)
  ATom_points.append(ATom_O3.loc[timesteps[i]])

# For this example, ATom J rate measurements are around e-5 but UKCA J rates are around e-11. 
# Find out if this is common in the data. 
# Check the sources ATom and Fast-J use for x-sections and quantum yields and see if they cause this. 
# compare ATom days with each other for O3.
# compare UM days with each other for O3.
# compare the ATom days with the UM days.
# put all the above into functions to work with different chemicals and times etc and try with NO2.
# compare ratios of other rates to NO2 at same points for both UCKA and ATom.

#outpath = './test_cube.nc'
#iris.save(UKCA_O3, outpath)

