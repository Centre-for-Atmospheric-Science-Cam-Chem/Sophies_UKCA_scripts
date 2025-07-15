'''
Name: Sophie Turner.
Date: 3/10/2024.
Contact: st838@cam.ac.uk
Reduce the size of a whole year of UKCA output data into a dataset small enough to fit in < 400 GB memory.
'''

import time
import glob
import numpy as np
import constants as con
import functions as fns
import file_paths as paths


def sample_day(i, year_files, points, data_new):
  # Takes ~ 1 minute per day file for 2 million points.
  print(f'Processing file {i+1} of {len(year_files)}.')
  # Get this file.
  day_file = year_files[i]
  data = np.load(day_file)
  # Remove night and upper portion where Fast-J might not work. 
  #data = fns.day_trop(data)
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
points = 8700000

# Prepare the new file and data array of 32 bit floats.
name_new = 'full_range_2y'
path_data_new = f'{paths.npy}/{name_new}.npy'
path_meta_new = f'{paths.npy}/{name_new}_metadata.txt'
data_new = np.empty((con.n_fields, 0), dtype=np.float32)

# Get the year of npy files.
year_files = glob.glob(f'{paths.npy}/201?????.npy')

# Every day file...
for i in range(len(year_files)):
  data_new = sample_day(i, year_files, points, data_new)

'''
# 2 consecutive in every 8 files...
for i in range(0, len(year_files), 8):
  data_new = sample_day(i, year_files, points, data_new)
  next = i + 1
  # If this is not the last file, get the next day file and do the same with it.
  if next < len(year_files):
    data_new = sample_day(next, year_files, points, data_new)

'''
# Write & save metadata about the dataset.
text = f'The dataset, {name_new}.npy, is 21 days, over 2 years, of UM output data at a resolution low enough to fit in < 400 GB program memory.\n\
Each day of data contains {points} hourly grid points (reduced from 56.4 million), randomly sampled from the full, unchanged range of data.'
print(f'Writing metadata to {path_meta_new}')
meta = open(path_meta_new, 'w')
meta.write(text)
meta.close()  

# Save the new dataset.
print(f'Saving new dataset at {path_data_new}')
np.save(path_data_new, data_new)
