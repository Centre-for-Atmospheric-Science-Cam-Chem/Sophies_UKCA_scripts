'''
Date: 20/7/2024
Contact: st838@cam.ac.uk
Verify the integrity of data in original .pp, flattened .npy and pytorch formats.
Check that rows/cols of data remain unchanged and in the correct order. 
To run on JASMIN Sci computers.
'''

import os
import cf
import torch
import numpy as np
import file_paths as path

# Data dimensions.
times = 24
alts = 85
lats = 144
lons = 192

# Pick a date.
date = '20151015'

# Load original .pp file.
pp = f'{path.data}/cy731a.pl{date}.pp'
data_pp = cf.read(pp) # 16 GB.
print('\npp list:\n', data_pp)
print('\npp data:', len(data_pp), data_pp[0].shape)

# Load flattened .npy file.
npy = f'{path.data}/{date}.npy'
data_npy = np.load(npy) # 18 GB.
print('\nnp data:', data_npy.shape)

# Follow steps taken for NNs when making tensor.
data_torch = np.swapaxes(data_npy, 0, 1) # 18 GB.

# Make Pytorch tensor.
data_torch = torch.from_numpy(data_torch.copy()) # 18 GB.
print('\npytorch data:', data_torch.shape)

# Pick a data point dims.
# For converting between indexing systems of different datasets.
item = 78 # STASH item.
time = 3
alt = 15
lat = 38
lon = 25

# Pick a data point in .pp.
point_pp = data_pp[item].array
point_pp = point_pp[time, alt, lat, lon]
print('\npoint pp:', point_pp)

# Convert index 0 to different structure used in .npys.
if item <= 8:
  i = item + 5
elif item > 12:
  i = item + 1
  
# Convert other indices to flattened format, place-wise.
j0 = time * alts * lats * lons
j1 = alt * lats * lons
j2 = lat * lons
j3 = lon
j = j0 + j1 + j2 + j3

# Find the same data points in .npy.
point_npy = data_npy[i, j]
print('\npoint npy:', point_npy)

# Find the same data points in tensor.
point_torch = data_torch[j, i]
print('\npoint torch:', point_torch)
