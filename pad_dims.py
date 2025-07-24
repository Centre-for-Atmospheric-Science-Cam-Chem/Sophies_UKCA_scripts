'''
Name: Sophie Turner.
Date: 8/3/2024.
Contact: st838@cam.ac.uk
Compile data from daily UM .pp files and make a big .npy file containing all UKCA dimensions
flattened. Time and space are added as columns. 
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
import file_paths as paths


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
  print(f'That took {round(minutes, 1)} minutes.')
  return(cols)

 
# Input files. 
ukca_files = glob.glob(f'{paths.pp}/cy731a.pl*.pp') # Any .pp file will do as long as it's a day. 
# Output file path for padded dims to match flattened fields.
out_path = f'{paths.npy}/dims.npy'

print('Reading the .pp file')
field = cf.read(ukca_files[0], select='stash_code=50500')[0]

# Get the dimension values.
print('Getting the dimension values as np arrays.')
hours = field.coord('time').hour.array # Hour of day.
alts = field.coord('atmosphere_hybrid_height_coordinate').array
lats = field.coord('latitude').array
lons = field.coord('longitude').array

nhours = len(hours)
nalts = len(alts)
nlats = len(lats)
nlons = len(lons)

stride_hour = nalts * nlats * nlons 
stride_alt = nlats * nlons 
stride_lat = nlons
stride_lon = 1

rep_hour = 1
rep_alt = nhours
rep_lat = nhours * nalts
rep_lon = nhours * nalts * nlats
  
full_size = nhours * nalts * nlats * nlons  
  
cols = np.empty((0,full_size))   
print('Padding flat hours.')
cols = pad_dim(hours, rep_hour, stride_hour, cols)
print('Padding flat altitude.')
cols = pad_dim(alts, rep_alt, stride_alt, cols)
print('Padding flat latitude.')
cols = pad_dim(lats, rep_lat, stride_lat, cols)
print('Padding flat longitude.')
cols = pad_dim(lons, rep_lon, stride_lon, cols)

print(f'Saving entire {cols.shape} array as {out_path}.')
np.save(out_path, cols)
