'''
Name: Sophie Turner.
Date: 3/10/2024.
Contact: st838@cam.ac.uk
Reduce the size of a whole year of UKCA output data into a dataset small enough to fit in < 400 GB memory.
'''

import os
import time
import glob
import math
import numpy as np
import file_paths as paths

# Global constant RNG.
RNG = np.random.default_rng(6)


def sample_day(i, year_files, points, data_new):
  # Takes ~ 6 hours on sci ph computers for 182 million samples.
  print(f'Processing file {i+1} of {len(year_files)}.')
  # Get this file.
  day_file = year_files[i]
  data = np.load(day_file)
  # Remove night and upper portion where Fast-J might not work.
  data = data[:, data[10] > 0] # Day.
  #data = data[:, data[7] > 20] # Trop.
  # Reduce data randomly. Seems naiive but is suitable due to large amount of data.
  full_size = len(data[0])
  ids = RNG.integers(0, full_size, points)
  ids = np.sort(ids)
  data = data[:, ids]
  # Add the data to the array of the new dataset.
  data_new = np.hstack((data_new, data))
  # Check the size of the new dataset.
  print('New dataset so far:', data_new.shape)
  return(data_new)


def sample_day_night(i, year_files, points, data_new):
  # Takes ~ 17.5 hours on sci ph computers for 182 million samples.
  print(f'Processing file {i+1} of {len(year_files)}.')
  # Get this file.
  day_file = year_files[i]
  data = np.load(day_file)
  # Split into day and night portions.
  day_data = data[:, data[10] > 0]
  night_data = data[:, data[10] == 0]
  day_size = math.floor(points * 0.99)
  day_ids = RNG.integers(0, len(day_data[0]), day_size)
  day_ids = np.sort(day_ids)
  day_data = day_data[:, day_ids]
  # Sample 1% of the data from night points.
  night_size = math.floor(points * 0.01)
  night_ids = RNG.integers(0, len(night_data[0]), night_size)
  night_idx = np.sort(night_ids)
  night_data = night_data[:, night_ids]
  # Stick the samples back together.
  # The night and day prtions will not be in order. Probably doesn't matter?
  data_new = np.hstack((data_new, day_data, night_data))
  return(data_new)


# How many data points we want per chosen day of data.
points = 500000

# Prepare the new file and data array of 32 bit floats.
name_new = 'day_strat_yr_182m'
path_data_new = f'{paths.data}/{name_new}.npy'
path_meta_new = f'{paths.data}/{name_new}_metadata.txt'
data_new = np.empty((85, 0), dtype=np.float32)

# Check if we are accidentally overwriting something.
if os.path.exists(path_data_new):
  print(f'Warning: The file {path_data_new} already exists.')
  overwrite = input('Do you want to overwrite the data in this file? (y/n): ').strip().lower()
  if overwrite != 'y':
    exit(1)

# Get the daily npy files for the year.
year_files = sorted(glob.glob(f'{paths.data}/2015????.npy'))
# Every day file...
for i in range(len(year_files)):
  start = time.time()
  data_new = sample_day(i, year_files, points, data_new)
  end = time.time()
  elapsed = end - start
  remaining = elapsed * (len(year_files) - (i + 1))
  minutes = remaining / 60
  print(f'Approximately {minutes} minutes remaining.')

# Write & save metadata about the dataset.
text = f'The dataset, {name_new}.npy, is a year of UM output data at a resolution low enough to fit in < 400 GB program memory.\n\
Each day of data contains {points} hourly grid points (reduced from 56.4 million), sampled from daytime data at all pressures.'
print(f'Writing metadata to {path_meta_new}')
meta = open(path_meta_new, 'w')
meta.write(text)
meta.close()

# Save the new dataset.
print(f'Saving new dataset at {path_data_new}')
np.save(path_data_new, data_new)
