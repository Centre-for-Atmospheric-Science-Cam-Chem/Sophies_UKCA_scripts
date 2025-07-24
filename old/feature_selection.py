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

import math
import glob
import numpy as np
import prediction_fns_numpy as fns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
  
  
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
  # Don't use this function if the field is empty/all zeros.
  # Find the name of a row.
  # row: a 1d npy array from an input set.
  # day: the 2d npy array containing all the data.
  # names: the array of processed metadata of row names.
  # Get the index of the row in the full dataset.
  if np.all(row == 0):
    name = 'all zero'
  else:
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
  input_rows = np.swapaxes(input_rows, 0, 1)
  return(input_rows)
 
 
def specific_num_inputs(inputs, targets, days, idx_names):
  # Test out a specific number of inputs.
  # Choose how many input rows to use. 
  num_inputs = 5

  # Do feature selection and dimensionality reduction if there are more potential inputs.
  if inputs.shape[1] > num_inputs:
    inputs = simple_select(inputs, targets, num_inputs) 
  
  print('Inputs chosen:')
  for i in range(num_inputs):
    name = get_name(inputs[:,i], days, idx_names)
    print(name) 

  # Reshape the targets so they enter ML functions in the right order.
  if targets.shape[0] < targets.shape[1]:
    targets = np.swapaxes(targets, 0, 1)
  
  # Split data (almost randomly).
  in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1, random_state=6)
  
  # Test to see if my old linear regression code will work with these data.
  model = fns.train(in_train, out_train)
  pred, err, r2 = fns.test(model, in_test, out_test)
  print('R2:', r2)
  print('Err:', err)
  
  
def try_many_inputs(inputs, targets, days, idx_names):
  # Test out different numbers of inputs in a loop.
  targets_rot = np.swapaxes(targets, 0, 1)

  # Test different numbers of input rows to see how many to use.
  #for num_inputs in range(6, 0, -1):
  num_inputs = 3
  print('\nNum inputs:', num_inputs)
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
  pred, mse, mape, r2 = fns.test(model, in_test, out_test)
  print('R2:', r2)
  print('MSE:', mse)

  
# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
input_files = glob.glob(f'{dir_path}/2015*.npy')
name_file = f'{dir_path}/idx_names'
training_data_path = f'{dir_path}/4days.npy'

# Names of the fields. See metadata.
idx_names = fns.get_idx_names(name_file)

# Get the training data.
days = fns.collate(training_data_path, input_files) 

# Have a look at the fields to choose which ones to use.  
for name in idx_names:
  print(name)
  
# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(0,15, dtype=int)
J_all = np.arange(15,85, dtype=int)
# J rates which are not summed or duplicate fg.
J_core = [16,18,19,20,24,25,28,29,30,31,32,33,34,36,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82]
# The above with usually zero rates removed.
J_core = [16,18,19,20,24,25,28,30,31,32,33,51,52,66,68,70,71,72,73,74,75,76,78,79,80,81,82]
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.

# Cut the higher altitudes off where Fast-JX might not work properly.
# 25km / 85km = 29% 
top = 0.29642513
days = days[:, np.where(days[1] <= top)].squeeze()

for target in J_core:
  print('\nTarget:', target)
  
  inputs = days[J_core]
  targets = days[target]

  # Reshape the arrays so they enter ML functions in the right order however many fields are selected.
  inputs = fns.shape(inputs)
  targets = fns.shape(targets)

  # Choose which set to remove duplicated row from.
  # Swap these around to remove from the other set.

  # Don't let target fields into inputs.
  # Usually if there are more input features than target features.
  inputs, targets = remove_duplicates(inputs, targets)

  # Don't let input features into targets.
  # Usually if there are more target features than input features.
  #targets, inputs = remove_duplicates(targets, inputs)

  # Reshape the inputs so they enter ML functions in the right order.
  inputs = np.swapaxes(inputs, 0, 1)

  try_many_inputs(inputs, targets, days, idx_names)
