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

 
def write_dims(field, start, out_dir):
  # field: a cf field from an opened .pp file.
  # start: index of time step to start at (in case the set was partially completed and needs finishing). 
  # Dimensions will be added as vectors.
  print('Flattening dimensions.')
  # Get the dimension values.
  dts = field.coord('time')    
  alts = field.coord('atmosphere_hybrid_height_coordinate')
  lats = field.coord('latitude')
  lons = field.coord('longitude')
  # For each entry in the field, set the time, alt, lat and long.
  # e.g. there should be 256 values if there are 4 values repeated 4x4x4x.
  # Repeat the dims in the right order to match the flattened data.
  for ts in range(start, dts.size):
    dt = dts[ts]
    s = time.time()
    cols = np.empty((4, 0), dtype = np.float32)
    print(f'Flattening 4d timestep {ts + 1} of {dts.size}.')
    for lvl in range(alts.size):
      alt = alts[lvl]
      print(f'Flattening along altitude level {lvl + 1} of {alts.size}.')
      for lat in lats:
        for lon in lons: 
          # Save these dimensions in their own columns.
	  # Times are converted to numerical representation.
          cols = np.append(cols, [dt, alt, lat, lon], axis=1)
    print(f'Shape of array: {cols.shape}')
    out_path = f'{out_dir}/dims{ts + 1}.npy'
    print(f'Saving flattened dims in a .npy file, {out_path}.')
    np.save(out_path, cols)
    print(f'Saved up to time step {ts + 1}')
    e = time.time()
    t = (e - s) / 60
    if t > 120:
      t = t / 60
      unit = 'hours'
    else:
      unit = 'minutes'
    print(f'That timestep took {round(t, 2)} {unit}.')
  
  
def join_dims(dims_dir, out_dir):
  # Join up already flattened dims from hourly files to a daily file.  
  # dims_dir: path to directory containing all the hourly dims.
  # out_dir: output path for joined file.
  print('Joining up timesteps into 1 file.')
  cols = np.empty((4, 0), dtype = np.float32)
  for i in range(24):
    print(f'Adding array {i + 1}.')
    dim_file = f'{dims_dir}/dims{i + 1}.npy' # Have to specify the exact path to make it work in the right order.
    s = time.time()
    print(dim_file) # Check that they are read in the right order.
    dims = np.load(dim_file)
    cols = np.hstack((cols, dims), dtype = np.float32)
    print('shape of cols:', cols.shape)
    e = time.time()
    t = (e - s) / 60
    print(f'Adding array {i + 1} took {round(t, 1)} minutes.')
  out_path = f'{out_dir}/dims.npy' 
  print(f'Saving entire {cols.shape} array at {out_path}.')
  np.save(out_path, cols)    
    

# Base.
dir_path = '/scratch/st838/netscratch/ukca_npy' 
# Input files.
ukca_files = glob.glob(dir_path + '/*.pp') # Just .pp files. 

# Write the dims that haven't been written already.
start = 0
for i in range(24):
  dims_path = f'{dir_path}/dims{i + 1}.npy'
  if os.path.exists(dims_path):
    start = i + 1
  else:
    break   
    
if start != 0:
  print(f'The first {start} timesteps are already done.')

if start != 24:
  field = cf.read(ukca_files[0], select='stash_code=50500')[0]
  write_dims(field, start, dir_path)

# Join them all up.
join_dims(dir_path, dir_path)

# Delete hourly files.
print('Deleting all the hourly array files.')
dim_files = glob.glob(test_path + '/dims?*.npy')
for dim_file in dim_files:  
  os.remove(dim_file)
