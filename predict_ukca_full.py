'''
Name: Sophie Turner.
Date: 10/5/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import sys
import time
import glob
import psutil
import numpy as np
import prediction_fns_shared as fns
from sklearn.model_selection import train_test_split


def shape(rows):
  # Make single input fields 2d and arrange so that they fit with other parameters going into the ML functions.
  if rows.ndim == 1:
    rows = np.vstack((rows, ))
  rows = np.rot90(rows, 3)
  return(rows)
  


# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
input_files = glob.glob(f'{dir_path}/2015*.npy')
name_file = f'{dir_path}/idx_names'

# Memory usage limit.
max_mem = 75 # Percentage.

# Names of the fields. See metadata.txt.
lines = open(name_file, 'r')
idx_names = []
for line in lines:  
  words = line.replace('\n', '')
  words = words.split(maxsplit=1) 
  lst = [words[0], words[1]]
  idx_names.append(lst)

'''
# Collate the data.
days = np.empty((83, 0))
for i in range(len(input_files)):
  print(f'Adding day {i+1} to array.')
  start = time.time()
  npy_file = input_files[i]
  data = np.load(npy_file)
  days = np.append(days, data, axis=1)
  mem_used = psutil.virtual_memory().percent
  if mem_used > max_mem:
    sys.exit(f'The maximum memory usage limit of {max_mem}% was exceeded with a {i+1} day array of size {days.shape}.')
  end = time.time()
  seconds = round(end - start)
  print(f'That day took {seconds} seconds.')
  print(f'Size of array: {days.shape}')
  print(f'Memory usage at {mem_used}%.')
'''

# TEST
# Sample for fast testing.
days = np.load(input_files[0])

# Have a look at the fields to choose which ones to use.  
#for name in idx_names:
#  print(name)
  
# Indices of some common combinations to use as inputs and outputs.
phys_all = np.linspace(0,12,12, dtype=int)
J_all = np.linspace(13,82,69, dtype=int)
H2O2 = 72
NO2 = 14
  
# Choose which fields to use as inputs and outputs. 
inputs = days[phys_all]
targets = days[H2O2]

# Reshape the arrays so they enter ML functions in the right order however many fields are selected.
inputs = shape(inputs)
targets = shape(targets)

# Split data (almost randomly).
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1, random_state=6)
  
# Test to see if my old linear regression code will work with these data.
model = fns.train(in_train, out_train)
pred, mse, r2 = fns.test(model, in_test, out_test)
print()
print(r2)
print()
