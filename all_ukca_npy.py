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

import os
import cf
import time
import glob
import numpy as np

def write_metadata(day, out_path):
  # Function to write metadata file.
  # day: an opened .pp file as a cfpython FieldsList.
  # out_path: the metadata output file path.
  # Identities of all 68 of the J rates output from Strat-Trop + some physics outputs.
  import codes_to_names as codes  
  code_names = np.array(codes.code_names)
  # Adjust them so that they work with cf identity functions.
  for i in range(len(code_names)):
    code = code_names[i,0]
    if code[0:2] == 'UM':
      code = f'id%{code}'
      code_names[i,0] = code
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
      i -= 1
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
  print('Flattening dimensions.')
  # Get the dimension values.
  dts = field.coord('time')    
  alts = field.coord('atmosphere_hybrid_height_coordinate')
  lats = field.coord('latitude')
  lons = field.coord('longitude')
  # For each entry in the field, set the time, alt, lat and long.
  # There should be 256 values if there are 4 values repeated 4x4x4x.
  # Repeat the dims in the right order to match the flattened data.
  table = np.empty((4, 0), dtype=np.float32)
  for dt in dts:
    for alt in alts:
      for lat in lats:
        for lon in lons: 
          # Save these dimensions in their own columns.
	  # Times are converted to numerical representation.
          table = np.append(table, [dt, alt, lat, lon], axis=1)
  np.save(out_path, table)
  return(table)
  

# Base.
dir_path = '/scratch/st838/netscratch/ukca_npy/' 
# Input files.
ukca_files = glob.glob(dir_path + '/*.pp') # Just .pp files. 
# Output paths.
meta_file = dir_path + 'metadata.txt'
npy_dims_file = dir_path + 'dims.npy' # Flattened dims to re-use.

# True if 1 npy file for each day of data, False if 1 npy file for all the data.
npy_day = True

# Write the dims first if they haven't been written already.
if os.path.exists(npy_dims_file):
  dims = np.load(npy_dims_file)
else:
  field = cf.read(ukca_files[0], select='stash_code=50500')[0]
  dims = write_dims(field, npy_dims_file)
print('shape of dims:', dims.shape)
if not npy_day:
  # We might need to build up the big np array in sections depending on the amount of data.
  if os.path.exists(npy_file):
    table = np.load(npy_file)
  else:
    table = dims
  del(dims) # There is no need to re-use the dims if making 1 big file.  

# Pick the day file to read.
for fi in range(len(ukca_files)):
  ukca_file = ukca_files[fi]
  print(f'Reading .pp file {fi+1} of {len(ukca_files)}:')
  print(ukca_file)
  day = cf.read(ukca_file)

  # Write metadata if needed.
  if not os.path.exists(meta_file):
    print(f'Writing metadata to {meta_file}.')
    write_metadata(day, meta_file)

  # Get the number of vertical levels.
  field = day[0]
  nalts = field.coord('atmosphere_hybrid_height_coordinate').size

  if npy_day:
    table = dims.copy() # Re-use the dims if making multiple files.   

  # For each field, save all the field data in another column.
  for i in range(len(day)):
    start = time.time()
    print(f'Converting field {i+1} of {len(day)}.')
    field = day[i]
    # We don't want section 30 pressure level outputs here.
    if field.identity()[:13] == 'id%UM_m01s30i':
      continue    
    # If there are 3 dimensions add a 4th and pad it so that the data match the flattened dims.
    if field.ndim == 3:    
      field = field.array
      times = nalts
      stride = field.shape[1] * field.shape[2]
      field = field.flatten()
      # Make a new 1d field array.
      padded = np.empty(0, dtype=np.float32)
      # Loop through old 1d field indices with strides of nlat x nlon until the end.
      for i in range(0, len(field), stride):
        # Pick sub arrays in sections of len nlat x nlon
        rep = field[i : i + stride]
        # Repeat that sub array ntime x nalt.
        rep = np.tile(rep, times)
        # Append it to the new field array.
        padded = np.append(padded, rep)
      field = padded
    else:
      field = field.flatten()   
    # Since cf python is so slow with its own flattened arrays, it's faster to convert the flat array to np
    # before working with it, depending on the size.
    # Don't convert to np unless you specifically need to because the conversion takes a long time.
    # This should probably be done on a GPU.  
    table = np.vstack((table, field), dtype=np.float32)
    print('shape of table:', table.shape)
    end = time.time()
    elapsed = end - start
    print(f'That field took {round(elapsed / 1)} seconds.')
  
  # Save the np array as a npy file containing this day of data.
  if npy_day:
    npy_file = dir_path + ukca_file[9:17] + '.npy'
    np.save(npy_file, table)
  
# Save the np array as a npy file containing all data.
if not npy_day:
  npy_file = dir_path + 'all.npy' 
  np.save(npy_file, table)
