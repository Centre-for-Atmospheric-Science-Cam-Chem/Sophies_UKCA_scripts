'''
Name: Sophie Turner.
Date: 8/3/2024.
Contact: st838@cam.ac.uk
Compile data from daily UM .pp files and make a big .npy file containing all UKCA dimensions
flattened. Time and space are added as columns. 
Each time step is saved separately and then they are joined after.
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import os
import cf
import time
import glob
import numpy as np


def pad_dim(dim, rep, stride, cols):
  start = time.time()
  padded = np.empty(0)  
  for i in range(rep):
    new_dim = np.repeat(dim, stride)
    padded = np.append(padded, new_dim)
  cols = np.r_[cols, [padded]] 
  del(new_dim, padded) 
  end = time.time()
  minutes = (end - start) / 60
  print(f'That took {round(minutes)} minutes.')
  return(cols)


# Base.
dir_path = '/scratch/st838/netscratch/ukca_npy' 
# Input files.
ukca_files = glob.glob(dir_path + '/*.pp') # Just .pp files. 
# Output file path for padded dims to match flattened fields.
out_path = f'{dir_path}/dims.npy'

# TEST
out_path = '/scratch/st838/netscratch/tests/dims.npy' 

print('Reading the .pp file')
field = cf.read(ukca_files[0], select='stash_code=50500')[0]

# TEST
field = field[:20, :20, :20, :20]

# Get the dimension values.
print('Getting the dimension values as np arrays.')
times = field.coord('time').array    
alts = field.coord('atmosphere_hybrid_height_coordinate').array
lats = field.coord('latitude').array
lons = field.coord('longitude').array

ntimes = len(times)
nalts = len(alts)
nlats = len(lats)
nlons = len(lons)

stride_time = nalts * nlats * nlons 
stride_alt = nlats * nlons 
stride_lat = nlons
stride_lon = 1

rep_time = 1
rep_alt = ntimes
rep_lat = ntimes * nalts
rep_lon = ntimes * nalts * nlats
  
full_size = ntimes * nalts * nlats * nlons  
  
cols = np.empty((0,full_size))   
print('Padding flat time.')
cols = pad_dim(times, rep_time, stride_time, cols)
print('Padding flat altitude.')
cols = pad_dim(alts, rep_alt, stride_alt, cols)
print('Padding flat latitude.')
cols = pad_dim(lats, rep_lat, stride_lat, cols)
print('Padding flat longitude.')
cols = pad_dim(lons, rep_lon, stride_lon, cols)

print(f'Saving entire {cols.shape} array as {out_path}.')
np.save(out_path, cols)
