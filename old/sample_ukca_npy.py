'''
Name: Sophie Turner.
Date: 8/3/2024.
Contact: st838@cam.ac.uk
Compile data from daily UM .pp files and make a big .npy file containing all UKCA data 
flattened with time and space added as columns. Takes a long time to run.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

print('Importing modules.')

import os
import cf
import time
import numpy as np
import pandas as pd
import codes_to_names as codes


def write_metadata(day, out_path):
  # Function to write metadata file.
  # day: an opened .pp file as a cfpython FieldsList.
  # out_path: the metadata output file path.
  # Make a file to write metadata to.
  meta_file = open(out_path, 'w')
  metadata = 'These are the ordered indices and column names of the numpy arrays in the .npy files in this directory.\
  \nSTASH code identities are included for each field where possible. \
  \nDate-times have been converted to numerical interpretations of time. \
  \nThe date-times which they represent are shown here, in the same order as the numpy data.\n\
  \nDate-times:\n'
  # Get date-times.
  for timestep in day[0]:
    metadata += f"{timestep.coord('time').data}\n"
  metadata += '\nColumn names:\n'
  # Add the time, alt, lat and long column names and units.
  metadata += '0 time\n1 altitude m\n2 latitude deg N\n3 longitude deg E\n'
  # Get the code-name of each field and write it in metadata.
  i = 0
  for field in day:
    code = field.identity()
    # We don't want section 30 pressure level outputs here.
    if code[:13] == 'id%UM_m01s30i':
      continue
    i += 1
    idx = np.where(code_names[:,0] == code)
    name = code_names[idx, 1][0][0]
    metadata += f'{i+4} {code} {name}\n'
  # Write all this to the file.
  meta_file.write(metadata)
  meta_file.close()
   

def write_dims(field, out_path):
  # field: a cf field from an opened .pp file. 
  # Dimensions will be added as columns.
  # It will be a 5d np array of (num fields + 4 dims) x num times x num alts x num lats x num lons. 
  print('Flattening dimensions.')
  start = time.time()
  # Get the dimension values.
  dts = field.coord('time')
  alts = field.coord('atmosphere_hybrid_height_coordinate')
  lats = field.coord('latitude')
  lons = field.coord('longitude')
  # For each entry in the field, set the time, alt, lat and long.
  # There should be 256 values if there are 4 values repeated 4x4x4x.
  # Repeat the dims in the right order to match the flattened data.
  cols = np.empty((4, 0))
  for dt in dts:
    for alt in alts:
      for lat in lats:
        for lon in lons: 
          # Save these dimensions in their own columns.
          cols = np.append(cols, [dt, alt, lat, lon], axis=1)
  # Times are converted to numerical representation.
  end = time.time()
  elapsed = end - start
  print(f'That took {round(elapsed, 1)} seconds.')  
  return(cols)


# Base.
dir_path = '/scratch/st838/netscratch/'  
# Input .pp files.
test_file = dir_path + 'nudged_J_outputs_for_ATom/cy731a.pl20160729.pp' # Test file.
# Output paths.
meta_file = dir_path + 'tests/metadata.txt'
csv_file = dir_path + 'tests/cols.csv'
npy_file = dir_path + 'tests/cols.npy'

# Sample size of each dimension length.
sample_sizes = [2, 3, 4]
dim_len = sample_sizes[0]

# Identities of all 68 of the J rates output from Strat-Trop + some physics outputs.
code_names = np.array(codes.code_names)
# Adjust them so that they work with cf identity functions.
for i in range(len(code_names)):
  code = code_names[i,0]
  if code[0:2] == 'UM':
    code = f'id%{code}'
  code_names[i,0] = code

# Pick one day to test on.
print('Reading .pp file.')
day = cf.read(test_file)

# Write metadata if needed.
if not os.path.exists(meta_file):
  print('Writing metadata to {meta_file}.')
  write_metadata(day, meta_file)

field = day[0]

# TEST SAMPLE.
#field = field[:dim_len, :dim_len, :dim_len, :dim_len]
# Get the number of vertical levels.
nalts = field.coord('atmosphere_hybrid_height_coordinate').size

# TEST 
# How long does it take to convert the whole field to np in one go?
for dim_len in sample_sizes:
  arr = field[:dim_len, :dim_len, :dim_len, :dim_len]
  print('Converting.')
  start = time.time()
  arr = arr.array.copy()
  end = time.time()
  elapsed = end - start
  print(arr.shape)
  print(f'That conversion took {round(elapsed, 1)} seconds.\n')
exit()

# Write the dims first if they haven't been written already.
if not os.path.exists(npy_file):  
  cols = write_dims(field, npy_file)
else:  
  # Open the npy file.
  cols = np.load(npy_file)

# For each field, save all the field data in another column.

# TEST
# for i in range(len(day)):
for i in range(6):

  start = time.time()
  print(f'Converting field {i+1} of {len(day)}.')
  field = day[i]
  # We don't want section 30 pressure level outputs here.
  if field.identity()[:13] == 'id%UM_m01s30i':
    continue    
    # If there are 3 dimensions add a 4th and pad it so that the data match the flattened dims.
  if field.ndim == 3:  
    
    # TEST
    field = field[:dim_len, :dim_len, :dim_len]
    
    field = field.array
    times = nalts
    stride = field.shape[1] * field.shape[2]    
    field = field.flatten()
    # Make a new 1d field array.
    padded = np.empty(0)
    # Loop through old 1d field indices with strides of nlats x nlons until the end.
    for i in range(0, len(field), stride):
      # Pick sub arrays in sections of len nlats x nlons
      rep = field[i:i+stride]
      # Repeat that sub array ntimes x nalts.
      rep = np.tile(rep, times)
      # Append it to the new field array.
      padded = np.append(padded, rep)
    
    # TEST
    # Check that the new array is the right size and shape.
    print('padded.shape:', padded.shape) # should be 1d and the same as one of the following.
    
    field = padded
    
  else:
  
    # TEST SAMPLE.
    field = field[:dim_len, :dim_len, :dim_len, :dim_len]
  
    field = field.flatten()   
  
  # Since cf python is so slow with its own flattened arrays, it's faster to convert the flat array to np
  # before working with it, depending on the size.
  # Don't convert to np unless you specifically need to because the conversion takes a long time.
  # This should probably be done on a GPU.  
  cols = np.vstack((cols, field))
  print('cols.shape:', cols.shape)
  end = time.time()
  elapsed = end - start
  print(f'That field took {round(elapsed, 1)} seconds.')

# Save the np array as a npy file.
np.save(npy_file, cols)

# Test - view the sample in a csv to check it looks right.
df = pd.DataFrame(cols)
df.to_csv(csv_file)
