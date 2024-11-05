'''
Name: Sophie Turner.
Date: 3/10/2024.
Contact: st838@cam.ac.uk
Reduce the size of a whole year of UKCA output data into a dataset small enough to fit in < 400 GB memory.
'''

import time
import glob
import numpy as np
import file_paths as paths


def sample_day(i, year_files, points, data_new):
  # Takes ~ 1 minute per day file.
  print(f'Processing file {i+1} of {len(year_files)}.')
  # Get this file.
  day_file = year_files[i]
  data = np.load(day_file)
  # Remove night.
  data = data[:, np.where(data[10] > 0)].squeeze()
  # Remove upper portion where Fast-J might not work.
  data = data[:, np.where(data[7] > 20)].squeeze()  
  # Reduce data randomly. Seems naiive but is suitable due to large amount of data.
  full_size = len(data[0]) 
  ids = np.random.randint(0, full_size, points)
  ids = np.sort(ids)
  data = data[:, ids]
  # Add the data to the array of the new dataset.
  data_new = np.hstack((data_new, data))    
  # Check the size of the new dataset.
  print('New dataset so far:', data_new.shape)
  return(data_new)
  

# How many data points we want per chosen day of data.
points = 2000000 # 2 million.

# Prepare the new file and data array of 32 bit floats.
name_new = 'low_res_yr_2015'
path_data_new = f'{paths.data}/{name_new}.npy'
path_meta_new = f'{paths.data}/{name_new}_metadata.txt'
data_new = np.empty((85, 0), dtype=np.float32)

# Get the year of npy files.
year_files = glob.glob(f'{paths.data}/2015*.npy')

# Every 8 files...
for i in range(0, len(year_files), 8):
  data_new = sample_day(i, year_files, points, data_new)
  next = i + 1
  # If this is not the last file, get the next day file and do the same with it.
  if next < len(year_files):
    data_new = sample_day(next, year_files, points, data_new)

# Write & save metadata about the dataset.
text = f'The dataset, {name_new}.npy, is a year of UM output data at a resolution low enough to fit in < 400 GB program memory.\n\
It is made of 2 consecutive days in every 8, from a UM simulation of 2015.\n\
Each day of data contains {points} hourly grid points (reduced from 56.4 million), randomly sampled from day-time data at pressures > 20 Pa.'
print(f'Writing metadata to {path_meta_new}')
meta = open(path_meta_new, 'w')
meta.write(text)
meta.close()  

# Save the new dataset.
print(f'Saving new dataset at {path_data_new}')
np.save(path_data_new, data_new)
