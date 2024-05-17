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
import math
import glob
import psutil
import numpy as np
import prediction_fns_shared as fns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression


def shape(rows):
  # Make single input fields 2d so that they fit with other parameters going into the ML functions.
  # rows: np array of fields chosen for training or testing.
  if rows.ndim == 1:
    rows = np.vstack((rows, ))
  return(rows)
  
  
def remove_duplicates(remove, keep):
  # Don't let targets into inputs. 
  # remove: np array of fields chosen for testing or training, from which duplicates will be removed.
  # keep: np array of fields chosen for testing or training, from which duplicates will be retained.
  removes = []
  for i in range(len(remove)):
    a1 = remove[i]
    for a2 in keep:
      if np.all(a1 == a2):
        removes.append(i)
  remove = np.delete(remove, removes, axis=0)
  return(remove, keep)
  

def get_name(row, day, names):
  # Find the name of a row.
  # row: a 1d npy array from an input set.
  # day: the 2d npy array containing all the data.
  # names: the array of processed metadata of row names.
  # Get the index of the row in the full dataset.
  for i in range(len(day)):
    each_row = day[i]
    if np.all(each_row == row):
      idx = i
      break
  # Match that index to the line of text in the metadata. 
  name = names[i][1]
  return(name)
  
  
def simple_select(inputs, targets, n):
  # Feature selection assuming that all targets will have the same best input rows.  
  target = targets[0]
  inputs = SelectKBest(f_regression, k=n).fit_transform(inputs, target)
  return(inputs)
  
  
def multi_target_select(inputs, targets, n):
  # Separate feature selection for multiple target fields.
  # Perform feature selection and dim reduction, including the best input rows for each target row.
  # inputs: np array of input rows from which to choose the best.
  # targets: np array of >1 target rows for which to find best intputs.
  # n: int specifying the desired number of input rows.
  # Choose k based on n but not constrained by it.
  # The function is deliberately flexible with this number to allow the best combination of inputs, 
  # so the number of input rows selected may not = n.
  num_targets = len(targets)
  k = math.ceil(n / num_targets) 
  input_rows = []
  for target in targets:
    add_row = SelectKBest(f_regression, k=k).fit_transform(inputs, target)
    if len(input_rows) == 0:
      input_rows.append(add_row)
    else:
      # If the best input feature is the same for multiple targets, don't duplicate it in input set.
      present = False
      for row in input_rows:
        if np.all(row == add_row):
          present = True
          break
        if not present:
          input_rows.append(add_row)
    if len(input_rows) >= n:
      break
  input_rows = np.array(input_rows)
  input_rows = input_rows[:,:,0]
  # Reshape the inputs so they enter ML functions in the right order.
  input_rows = np.rot90(input_rows, 3)
  return(input_rows)
 
 
def specific_num_inputs(inputs, targets, days, idx_names):
  # Test out a specific number of inputs.
  # Choose how many input rows to use. 
  num_inputs = 4

  # Do feature selection and dimensionality reduction if there are more potential inputs.
  if inputs.shape[1] > num_inputs:
    inputs = simple_select(inputs, targets, num_inputs) 
  
  print('Inputs chosen:')
  for i in range(num_inputs):
    name = get_name(inputs[:,i], days, idx_names)
    print(name) 

  # Reshape the targets so they enter ML functions in the right order.
  if targets.shape[0] < targets.shape[1]:
    targets = np.rot90(targets, 3)
  
  # Split data (almost randomly).
  in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1, random_state=6)
  
  # Test to see if my old linear regression code will work with these data.
  model = fns.train(in_train, out_train)
  pred, err, r2 = fns.test(model, in_test, out_test)
  print('R2:', r2)
  print('% err:', err)
  
  
def try_many_inputs(inputs, targets, days, idx_names):
  # Test out different numbers of inputs in a loop.
  targets_rot = np.rot90(targets, 3)

  # Test different numbers of input rows to see how many to use.
  for num_inputs in range(13, 0, -1): 
    print('Num inputs:', num_inputs)
    # Do feature selection and dimensionality reduction if there are more potential inputs.
    if inputs.shape[1] > num_inputs:
      inputs = simple_select(inputs, targets, num_inputs) 
  
    print('Inputs chosen:')
    for i in range(num_inputs):
      name = get_name(inputs[:,i], days, idx_names)
      print(name) 
  
    # Split data (almost randomly).
    in_train, in_test, out_train, out_test = train_test_split(inputs, targets_rot, test_size=0.1, random_state=6)
  
    # Test to see if my old linear regression code will work with these data.
    model = fns.train(in_train, out_train)
    pred, err, r2 = fns.test(model, in_test, out_test)
    print('R2:', r2)
    print('% err:', err)

  
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
phys_all = np.linspace(0,12,13, dtype=int)
J_all = np.linspace(13,82,70, dtype=int)
NO2 = 14
HCHOr = 16 # Radical product.
HCHOm = 17 # Molecular product.
H2O2 = 72
O3 = 76 # O(1D) product.
 
inputs = days[phys_all]
targets = days[H2O2]

# Reshape the arrays so they enter ML functions in the right order however many fields are selected.
inputs = shape(inputs)
targets = shape(targets)

# Don't let target rows into inputs.
# Choose which set to remove duplicated row from.
# Swap these around to remove from the other set.
inputs, targets = remove_duplicates(inputs, targets)

# Reshape the inputs so they enter ML functions in the right order.
inputs = np.rot90(inputs, 3)

specific_num_inputs(inputs, targets, days, idx_names)



