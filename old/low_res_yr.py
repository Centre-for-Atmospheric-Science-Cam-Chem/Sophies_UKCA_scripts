'''
Name: Sophie Turner.
Date: 10/4/2024.
Contact: st838@cam.ac.uk
Reduce the resolution of UM output data for ML model training.
'''

# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import sys
import glob
import psutil
import numpy as np
import matplotlib.pyplot as plt


def select_res(day, dim, res):
  # Select data by a specified resolution of a dimension.
  # day: all data from a daily .npy file. 
  # res: a divisor for reducing resolution by uniform strides.
  # dim: which flattened dimension's row to reduce. 0=time, 1=alt, 2=lat, 3=long.
  row = day[dim]
  # Get the indices of the cols which are at the points specified by resolution.
  row_set = np.unique(row)
  ids = np.empty(0, dtype=int)
  # Loop through the different values in order to find the right values.
  for i in range(res - 1, len(row_set), res):
    # Find their indices in the flattened row with dim duplicates.
    pos = np.where(row == row_set[i])
    ids = np.append(ids, pos)
  # Update selected data by removing parts we don't want.
  day = day[:, ids]
  return(day)


dir_path = '/scratch/st838/netscratch/ukca_npy' # Change this to the actual path.
# Directory with the npy files in for the whole year.
day_files = glob.glob(f'{dir_path}/*.npy') # Just full-sized .npy files.
out_dir = f'{dir_path}/reduced_dataset' # Where output will go.

# Test for night-time.
data_path = f'{dir_path}/20150115.npy'
print('Loading data.')
data = np.load(data_path)
print(data.shape)

print('Removing night times.')
data = data[:, np.where(data[10] > 0)].squeeze()
print('Removing upper stratosphere.')
data = data[:, np.where(data[7] > 20)].squeeze()
print(data.shape)

print('Taking a uniform random sub-sample.')
full_size = len(data[0]) 
new_size = 2000000 # 2 million.
i = np.random.randint(0, full_size, new_size)
sample = data[:, i]
print(sample.shape)

exit()

# Scale the sample up to the same size as the original data to compare their shapes.
sample_scaled = np.repeat(sample, 14, axis=1)
print(sample_scaled.shape)

# Plot them to see if the range/shape is still present.
plt.hist(sample[74], bins=100, histtype='step', label='Uniform random sub-sample')
plt.hist(sample_scaled[74], bins=100, histtype='step', label='Uniform random sub-sample, scaled up')
plt.hist(data[74], bins=100, histtype='step', label='Full dataset')
plt.title('Sub-sampling of JH2O2 with scaled-up sub-sample for distribution comparison')
plt.xlabel('J rate')
plt.ylabel('Number of data points')
plt.legend()
plt.show()

# Resolution definitions. Division by whole dataset. 
# 1 is all the data, 2 is half the data (every other point), etc.
# Original dims are 24 time x 85 alt x 144 lat x 192 lon = 56.4million points per day file.
# x 83 fields = 4.7billion elements per day file.

# Resolution for time (division, points=days, hours).
day_res = 1 # every day.
hr_res = 6 # 4 time steps per day. 
# Resolution for latitude (division, points).
lat_res = 4
# Resolution for longitude (division, points).
lon_res = 4

# Values for altitude (specific).
# Model levels 1, 12, 30, 40, 66.
# Surface, boundary layer, mid troposphere, tropopause, mid stratosphere.
alts_km = [0, 1, 6, 11, 30] # km.

# New dims are 4 time x 5 alt x 36 lat x 48 lon = 34,560 points per day file.
# x 83 fields = 2.9million elements per day file.

# Sizes in memory.
GB = 1000000000 # 1 GB in bytes.
# Choose what % of the directory's storage space we are willing to spend on output.
max_store = 95
# Choose what we want the maximum output file size to be.
max_out = GB * 50
# Choose what % of system memory we are willing to use up, excluding swap. 
max_mem = 95

# The new dataset.
days = np.empty(0)
days_size = 0

# Keep track of how many files are saved.
chunk = 0

# Depending on time resolution chosen, open npy files one by one.
for f in range(0, len(day_files), day_res):
  day_file = day_files[f] 
  day = np.load(day_file)

  # Check the size of the initial dataset.
  full_size = day.itemsize * day.size

  # Select by time.
  day = select_res(day, 0, hr_res)

  # Select by altitude. The altitudes chosen are not at uniform intervals so this is done separately.
  # Compare the real altitude values to the altitudes we want.
  # km = alt * 85. Find the closest to what we want.
  day[1] = day[1] * 85
  alts = day[1]
  i_cols = np.empty(0, dtype=int)
  for alt_km in alts_km:
    diffs = abs(alts - alt_km)
    idx_alt = diffs.argmin()
    i_alts = np.where(diffs == diffs.min())[0]
    i_cols = np.concatenate((i_cols, i_alts))
  i_cols = np.sort(i_cols)  

  # Update selected data by removing parts we don't want.
  day = day[:, i_cols]

  # Select by latitude.
  day = select_res(day, 2, lat_res)
  # Select by longitude.
  day = select_res(day, 3, lon_res)

  # Keep track of memory size of array.
  day_size = day.itemsize * day.size
  new_size = days_size + day_size
  res = full_size / day_size

  # Check what % of memory is used up.
  mem_used = psutil.virtual_memory().percent
  # Check what % of storage is used up. 
  store_used = psutil.disk_usage(out_dir).percent
  if mem_used <= max_mem and new_size <= max_out:
    # Concatenate the new array(s) if there is space.
    days = np.hstack(days, day)
    days_size = new_size
  elif new_size <= max_store: # Change this so that it actually checks the storage space.
    pass
  elif store_used <= max_store:
    # Save the new array(s) if there is space.
    chunk += 1
    out_path = f'{out_dir}/1in{res}_{chunk}.npy'
    np.save(out_path, days)
    days = np.empty(0)
    days_size = 0
  else:
    sys.exit(f"Dataset size exceeds limit. Dataset is {new_size} bytes. \
    {mem_used}% of the system memory was used and the limit is {max_mem}%. \
    {store_used}% of the output directory's storage space was used and the limit is {max_store}%.")
  
# Save the last array unless it has already been saved.
if days_size > 0:
  chunk += 1
  out_path = f'{out_dir}/1in{res}_{chunk}.npy'
  np.save(out_path, days)
 
