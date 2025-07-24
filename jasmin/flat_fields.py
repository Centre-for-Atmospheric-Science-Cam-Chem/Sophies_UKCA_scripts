'''
Name: Sophie Turner.
Date: 3/6/2024.
Contact: st838@cam.ac.uk
Compile data from daily UM .pp files and make a big .npy file containing all UKCA data flattened.
For use on JASMIN sci computers. 
Files are located at /gws/nopw/j04/um_ml_j_rates.
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
import file_paths as path


def write_metadata(day, out_path, dt=False):
  # Function to write metadata file.
  # day: an opened .pp file as a cfpython FieldsList.
  # out_path: the metadata output file path.
  # dt: bool whether to include examples of date-time representations.
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
  \nDate-times have been converted to numerical interpretations of time, as days since the start of the model run.\n'
  # If the data-times are to be included in metadata.
  if dt:
    metadata += 'A 1-day example of date-times which they represent are shown here, in the same order as the numpy data.\n\
    \nDate-times:\n'
    # Get date-times.
    for timestep in day[0]:
      metadata += f"{timestep.coord('time').data}\n"
  metadata += '\nColumn names:\n'
  # Add the time, alt, lat and long column names and units.
  metadata += '0 hour of day\n1 altitude m\n2 latitude deg N\n3 longitude deg E\n4 date-time\n'
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
  
  
def pad_time(field, rows):
  # Add 'days since' measurement as a row, in addition to the hour from dims.
  # field: any cf field from the necessary day.
  # rows: 2d numpy array of flattened data (to build up). 
  print('Padding time.')
  # Get the padding factors for time.
  times = field.coord('time').array # Days since start of run.
  nalts = field.coord('atmosphere_hybrid_height_coordinate').size
  nlats = field.coord('latitude').size
  nlons = field.coord('longitude').size
  stride = nalts * nlats * nlons 
  # Pad time so that it fits, and add to flat table.
  new_times = np.repeat(times, stride)
  rows = np.r_[rows, [new_times]] 
  del(new_times) 
  return(rows) 
  


# Input files.
ukca_files = glob.glob(path.data + '/*2015*.pp') # The new files of test data.
dims_file = path.data + '/dims.npy' # Dims to match the flattened data.
# Output paths.
meta_file = path.data + '/metadata.txt'
npy_file = path.data + '/fields.npy' # Flattened fields to re-use.

# True if 1 npy file for each day of data, False if 1 npy file for all the data.
npy_day = True

# If there are dims available, use them. If not, continue without them and add them later.
if os.path.exists(dims_file):
  dims = np.load(dims_file)
else:
  dims = np.empty(0, dtype = np.float32)

print(f'Shape of dims: {dims.shape}')

if not npy_day:
  # We might need to build up the big np array in sections depending on the amount of data.
  if os.path.exists(npy_file):
    table = np.load(npy_file)
    # Add the dims as the first 4 rows.
    table = np.vstack((dims, table))
  else:
    table = dims
  del(dims) # There is no need to re-use the dims if making 1 big file. 

process = False
# Pick the day file to read.
for fi in range(len(ukca_files)):

  # Specify start and end of files to process.
  start = '20151129' # Inclusive
  end = '20151130' # Exclusive

  # Only process the files that haven't been done yet.
  date = ukca_files[fi][-11:-3]  
  if date == start:
    process = True
  elif date == end:
    break
  if not process:
    continue

  start = time.time()
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

  # If making multiple files.
  if npy_day:
    table = dims.copy() # Re-use the dims if making multiple files. 

  # Add the absolute time (as 'days since start of run') as a row.
  table = pad_time(field, table)

  # For each field, save all the field data in another row.
  for i in range(len(day)):
    print(f'Converting field {i+1} of {len(day)}.')
    field = day[i]
    # We don't want section 30 pressure level outputs here.
    if field.identity()[:13] == 'id%UM_m01s30i':
      continue    
    # If there are only 3 dimensions add a 4th and pad it so that the data match the flattened dims.
    if field.ndim == 3:    
      field = field.array
      times = nalts
      stride = field.shape[1] * field.shape[2]
      field = field.flatten()
      # Make a new 1d field array.
      padded = np.empty(0, dtype=np.float32)
      # Loop through old 1d field indices with strides of nlat x nlon until the end.
      for i in range(0, len(field), stride):
        # Pick sub arrays in sections of len nlat x nlon.
        rep = field[i : i + stride]
        # Repeat that sub array ntime x nalt.
        rep = np.tile(rep, times)
        # Append it to the new field array.
        padded = np.append(padded, rep)
      field = padded
    else:
      field = field.flatten()     
    table = np.vstack((table, field), dtype=np.float32)
    print('Shape of table:', table.shape) 
  
  # Save the np array as a npy file containing this day of data.
  if npy_day:
    npy_file = f'{path.data}/{date}.npy'
    print('Saving .npy file.')
    np.save(npy_file, table)
    
  end = time.time()
  elapsed = end - start
  print(f'That file took {round(elapsed / 60)} minutes.') 
  
# Save the np array as a npy file containing all data.
if not npy_day:
  npy_file = dir_path + 'all.npy' 
  np.save(npy_file, table)
