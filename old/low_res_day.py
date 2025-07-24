'''
Name: Sophie Turner.
Date: 3/10/2024.
Contact: st838@cam.ac.uk
Reduce the size of a whole year of UKCA output data into a dataset small enough to fit in < 400 GB memory.
'''

import numpy as np
import file_paths as paths
import matplotlib.pyplot as plt

# How many data points we want per chosen day of data.
points = 50000

# Get the original file.
day_file = f'{paths.npy}/20150115.npy'

# Prepare the new file and data array of 32 bit floats.
name_new = 'random'
path_data_new = f'{paths.npy}/{name_new}.npy'

print('Loading data.')
data_orig = np.load(day_file)

#print(f'Reducing data from {data.shape} to (85, {points}).')  

# Remove night.
data = data_orig.copy()
data = data[:, np.where(data[10] > 0)].squeeze()
# Remove upper portion where Fast-J might not work.
data = data[:, np.where(data[7] > 20)].squeeze()  
# Reduce data.
full_size = len(data[0]) 

# Random.
ids = np.random.randint(0, full_size, points)
ids = np.sort(ids)

# Uniform.
#ids = np.linspace(0, full_size-1, points, dtype=np.int32)

data_new = data[:, ids]
print(data_new.shape)

div = round(full_size / points)
scaled = np.repeat(data_new, div, axis=1)

# Check distribution of important fields.
for i in [0,1,2,3,7,8,9,10,16]:
  print(i)
  spread = np.unique(data[i])
  if len(spread) < 50:
    bins = len(spread)
  else:
    bins = 50
  plt.hist(data_orig[i], bins=bins, histtype='step', label='All UM output')  
  plt.hist(data[i], bins=bins, histtype='step', label='After removing stratosphere and night')
  plt.hist(data_new[i], bins=bins, histtype='step', label='Random subsample') 
  plt.hist(scaled[i], bins=bins, histtype='step', label='Subsample, scaled up') 
  plt.legend()
  plt.show()
  plt.close()
