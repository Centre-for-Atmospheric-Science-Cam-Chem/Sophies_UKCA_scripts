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

import cf
import time
import numpy as np
import codes_to_names as codes
import prediction_fns_shared as fns


def write_metadata(day, out_path):
  # Function to write metadata file.
  # day: an opened pp file.
  # out_path: the metadata output file path.

  # Make a file to write metadata to.
  meta_file = open(out_path, 'w')
  metadata = 'These are the ordered indices and column names of the numpy arrays in the .npy files in this directory.\
  \nSTASH code identities are included for each field where possible.\n'
  # Add the time, alt, lat and long column names and units.
  metadata += '0 time s\n1 altitude m\n2 latitude deg N\n3 longitude deg E\n'

  # Get the code-name of each field and write it in metadata.
  for i in range(len(day)):
    field = day[i]
    code = field.identity()
    idx = np.where(code_names[:,0] == code)
    name = code_names[idx, 1][0][0]
    metadata += f'{i+4} {code} {name}\n'

  # Write all this to the file.
  meta_file.write(metadata)
  meta_file.close()


# Identities of all 68 of the J rates output from Strat-Trop + some physics outputs.
code_names = np.array(codes.code_names)
# Adjust them so that they work with cf identity functions.
for i in range(len(code_names)):
  code = code_names[i,0]
  if code[0:2] == 'UM':
    code = f'id%{code}'
  code_names[i,0] = code
  
# Input pp files.
dir_path = '/scratch/st838/netscratch/nudged_J_outputs_for_ATom/'
# Choose a day from each season.
spring_file = 0
summer_file = dir_path + 'cy731a.pl20160729.pp' # Test file.
autumn_file = 0
winter_file = 0
# Pick one day to test on.
day = cf.read(summer_file)

# Write metadata.
#write_metadata(day, '/scratch/st838/netscratch/tests/metadata.txt')

# Dimensions will be added as columns.
# It will be a 5d np array of (num fields + 4 dims) x num times x num alts x num lats x num lons. 

# This will become the big np array.
all_np = []

# Select the first field.
field = day[0]

# Take a smaller chunk of it to test with.
field = field[:4, :4, :4, :4]

# Converting between flattened and multi-dimensional C indices is the same as converting between 
# positional number systems where each dimension length is a base.
# So, in the case where each dim is of length 4, 
# 4d index 1,1,1,1 should = 1d index 1x4^(3) + 1x4^(2) + 1x4^(1) + 1x4^(0) = 85
# Here, the dims are in the order, time, alt, lat, long.
field = field.flatten()

# Since cf python is so slow with its own flattened arrays, it's faster to convert the flat array to np.
# Don't convert to np unless you specifically need to index things because the conversion takes a long time.
# This should probably be done on a GPU.
#field = field.data.array

# For each entry in the field, get the time, alt, lat and long.
# There should be 256 values, or 4 values repeated 4x4x4x.

# Convert time to seconds.
# Function to convert time to seconds is in a shared module.

# Save these dimensions in their own columns.

# Save the field entry value as well.

# For each field after the first, save all the field data in another column.

# Save the np array as a npy file.

# Close the pp file.
