'''
Name: Sophie Turner.
Date: 10/4/2024.
Contact: st838@cam.ac.uk
Reduce the resolution of UM output data for ML model training.
For use on JASMIN science computers. 
'''

# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import glob
import numpy as np

# Directory with the npy files in for the whole year.
day_files = glob.glob('*.npy') # Just .npy files. 

# Resolution definitions. Division by whole dataset. 
# 1 is all the data, 2 is half the data (every other point), etc.
# Original dims are 24 time x 85 alt x 144 lat x 192 lon = 56.4million points per day file.
# x 83 fields = 4.7billion elements per day file.

# Resolution for time (division, points=hours).
hr_res = 6 # 4 time steps per day. 

# Values for altitude (specific).
# Surface, boundary layer, mid troposphere, tropopause, mid stratosphere.
alts_km = [0, 1, 6, 11, 30] # km.
alts_lvl = [1, 12, 30, 40, 66] # model levels.

# Resolution for latitude (division, points).
lat_res = 4

# Resolution for longitude (division, points).
lon_res = 4

# New dims are 4 time x 5 alt x 36 lat x 48 lon = 34,560 points per day file.
# x 83 fields = 2.9million elements per day file.

# Sizes in memory.
GB = 1000000000 # 1 GB in bytes.
GB100 = GB * 100 # 100 GB.
# Choose how much storage space we are willing to spend on output array.
max_store = GB100 # 100 GB.
# Choose how much program memory we are willing to spend on np array. 
max_mem = GB100 # 100 GB.

# TEST
day = np.load('/scratch/st838/netscratch/tests/cols.npy') # Small test file.
print(day.shape)

'''
# Depending on time resolution chosen, open npy files one by one.
if hr_res > 24: # 24 hours per day file.
  pass # Open whichever day files we need.
else: # We need to open every day file.
  for day_file in day_files:
    day = np.load(day_file)
'''

# New array for selected data.
points = np.empty(0)

# Loop through the data to select the points we want. 

# Select by time.

# Select by altitude.

# TEST
alts_km = [0, 1]

# Compare the real altitude values to the altitudes we want.
# km = alt * 85. Find the closest to what we want.
day[1] = day[1] * 85
i_cols = np.empty(0, dtype=int)
for alt_km in alts_km:
  diffs = abs(day[1] - alt_km)
  idx_alt = diffs.argmin()
  i_alts = np.where(diffs == diffs.min())[0]
  i_cols = np.concatenate((i_cols, i_alts))
i_cols = np.sort(i_cols)  

# TEST
#print(i_cols)

# Update selected data by removing parts we don't want.
day = day[:, i_cols]
print(day.shape)

# Select by latitude.

# Select by longitude.

# Keep track of memory size of array.

# Save or concatenate the new array(s) if there is space.

# If there is not space to save the new array(s), go straight to ML.

# ML model.

