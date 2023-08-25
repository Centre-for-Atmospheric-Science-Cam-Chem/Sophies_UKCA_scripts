"""
Scripts to practise making ATom data comparable with UKCA data. 
Later on, this will be done properly with the SSP AMIP suites. 
Using my outputs from the test suite for now.

"""

import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This step takes several minutes. Better to load individual chunks than the whole thing. See stash codes text file.
p_file = './atmosa.pb1988sep'
#UKCA_data = iris.load(p_file)
UKCA_O3 = iris.load_cube(p_file,iris.AttributeConstraint(STASH='m01s50i567')) # O3 --> O(1D)

'''
print(type(UKCA_data))
print(UKCA_data.shape)
print(UKCA_data)

print(type(UKCA_O3)) # iris cube
print(UKCA_O3.shape) # 47 t, 85 z, 144 y, 192 x
UKCA_O3
UKCA_O3.data
'''

# The UKCA test suite simulates 1st and 2nd September 1988.
# The most similar time of year in the ATom data is 23rd August 2016.

# Open the .csv of all the ATom data which I have already pre-processed in the script, ATom_J_data.
ATom_file = '/content/drive/MyDrive/Documents/AI4ER/PhD/Photolysis_data/ATom_MER10_Dataset.20210613/photolysis_data.csv'
ATom_data = pd.read_csv(ATom_file, index_col=0) # shape = 138454, 49. 138454 time steps, 49 = chemicals + spatial dimensions.

# Pick out just the 22nd and 23rd August indices separately.
A_day1 = ATom_data[ATom_data.index.str.contains('2016-08-22')] # shape = 1994, 49. 1994 time steps for 22/8/16, 49 = chemicals + spatial dimensions.
A_day1_O3 = A_day1[['G_LAT', 'G_LONG', 'G_ALT', 'jO3_O2_O1D_CAFS']] # shape = 1994, 4.

A_day2 = ATom_data[ATom_data.index.str.contains('2016-08-23')] # shape = 1765, 49. 1765 time steps for 23/8/16, 49 = chemicals + spatial dimensions.
A_day2_O3 = A_day2[['G_LAT', 'G_LONG', 'G_ALT', 'jO3_O2_O1D_CAFS']] # shape = 1765, 4.

# Check what the time of day range is for both.
print(A_day1_O3.index[0]) # 12:53:40
print(A_day1_O3.index[-1], '\n') # 18:51:40
print(A_day2_O3.index[0]) # 15:05:40
print(A_day2_O3.index[-1]) # 20:00:20

# Trim so that all arrays are over the same times. # 15:05:40 to 18:51:40
A_day1_O3 = A_day1_O3.loc['2016-08-22 15:05:40':'2016-08-22 18:51:40'] # shape = 1205, 4. type = pandas dataframe.
A_day2_O3 = A_day2_O3.loc['2016-08-23 15:05:40':'2016-08-23 18:51:40'] # shape = 1353, 4. type = pandas dataframe.

# shape = 47 t, 85 z, 144 y, 192 x. type = iris cube.

# Check what time zones ATom and UM are using. ATom: UTC. UM: UTC.
# Get the UM data for the same time of day for both 1st and 2nd sep separately.
U_day1_O3 = UKCA_O3[13:18]
U_day2_O3 = UKCA_O3[37:42] # shape = 5 t, 85 z, 144 y, 192 x. type = iris cube.

times = ['2016-08-22 15:05:40', '2016-08-22 16:00:00', '2016-08-22 17:00:00', '2016-08-22 18:00:00', '2016-08-22 18:51:40']
U_day, A_day = [], []

for iTime in range(len(times)):

  t = U_day1_O3[iTime]

  # Check what units of altitude they're using. ATom: m. UM: m.
  # Find out what the altitudes are for ATom and select the right altitudes of UKCA data.
  Talt = A_day1_O3.loc[times[iTime]]['G_ALT']
  Ualts = t.coord('level_height').points
  diffs = np.absolute(Ualts-Talt)
  iAlt = diffs.argmin()
  t = U_day1_O3[iTime,iAlt]

  # Check what units of latitude they're using. ATom: degrees +N. UM: degrees.
  # Find out what the latitudes are for ATom and select the right latitudes of UKCA data.
  Tlat = A_day1_O3.loc[times[iTime]]['G_LAT']
  Ulats = t.coord('latitude').points
  diffs = np.absolute(Ulats-Tlat)
  iLat = diffs.argmin()
  t = U_day1_O3[iTime,iAlt,iLat]

  # Check what units of longitude they're using. ATom: degrees +E in half circles (-180 to 180). UM: degrees in a full circle (0 to 360).
  # Find out what the longtitudes are for ATom and select the right longitudes of UKCA data.
  Tlong = A_day1_O3.loc[times[iTime]]['G_LONG']
  Ulongs = t.coord('longitude').points
  if Tlong < 0:
    Tlong = Tlong + 360
  diffs = np.absolute(Ulongs-Tlong)
  iLong = diffs.argmin()
  t = U_day1_O3[iTime,iAlt,iLat,iLong]

  U_day.append(t)
  A_day.append(A_day1_O3.loc[times[iTime]])

print(U_day[0])
print(A_day[0])

# Used Monsoon to check if sigma is the J rate. sigma is not J rate.
# compare the 2 ATom days with each other for O3.
# compare the 2 UM days with each other for O3.
# compare the ATom days with the UM days.
# Now put all the above into functions to work with different chemicals and times etc and try with NO2.

#print(U_day[0].data)
#U_day1_O3.data[iTime,iAlt,iLat,iLong]
#print(UKCA_O3.data)

outpath = '/content/drive/MyDrive/Documents/AI4ER/PhD/Photolysis_data/test_cube.nc'
iris.save(UKCA_O3, outpath)
