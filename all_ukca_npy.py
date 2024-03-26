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
	  # Times are converted to numerical representation.
          cols = np.append(cols, [dt, alt, lat, lon], axis=1)
  return(cols)


# Base.
dir_path = '/scratch/st838/netscratch/ukca_npy/' 
# Input files.
ukca_files = glob.glob(dir_path + '/*.pp') # Just .pp files. 
# Output paths.
meta_file = dir_path + 'metadata.txt'
npy_file = dir_path + 'cols.npy'

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

  # Write the dims first if they haven't been written already.
  if not os.path.exists(npy_file):  
    cols = write_dims(field, npy_file)
  else:  
    # Open the npy file.
    cols = np.load(npy_file)

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
      padded = np.empty(0)
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
    cols = np.vstack((cols, field))
    end = time.time()
    elapsed = end - start
    print(f'That field took {round(elapsed / 60, 1)} minutes.')

# Save the np array as a npy file.
np.save(npy_file, cols)
