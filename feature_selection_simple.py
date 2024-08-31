'''
Name: Sophie Turner.
Date: 31/8/2024.
Contact: st838@cam.ac.uk
Find best UKCA inputs to predict UKCA J rates.
'''

import numpy as np
import prediction_fns_numpy as fns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression


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


# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
input_file = f'{dir_path}/20151015.npy'
name_file = f'{dir_path}/idx_names'

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(0,15, dtype=int)
J_all = np.arange(15,85, dtype=int)
# J rates which are not summed or duplicate fg.
J_core = [16,18,19,20,24,25,28,29,30,31,32,33,34,36,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82]
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.

# 2D list of indices and names of the fields. See metadata.
idx_names = fns.get_idx_names(name_file)

# The dataset.
day = np.load(input_file)

# Cut the higher altitudes off where Fast-JX might not work properly.
# 25km / 85km = 29% 
top = 0.29642513
day = day[:, np.where(day[1] <= top)].squeeze()

inputs = day[phys_all]
targets = day[O3] # Only 1 target feature allowed.
n_best = 2

if targets.ndim > 1:
  print('Please choose a single target feature.')
  exit()

# Reshape the arrays so they enter ML functions in the right order however many fields are selected.
inputs = fns.shape(inputs)
targets = fns.shape(targets)

# Remove fields which only contain zeros.
inputs = fns.remove_all_zero(inputs)
targets = fns.remove_all_zero(targets)

# Choose which set to remove duplicated row from.
# Swap these around to remove from the other set.
# Don't let target fields into inputs.
# Usually if there are more input features than target features.
inputs, targets = remove_duplicates(inputs, targets)

print('\nFinding the best input fields for predicting:')
for field in targets:
  name = get_name(field, day, idx_names)
  print(name)

# Reshape the data so they enter ML functions in the right order.
inputs = np.swapaxes(inputs, 0, 1)
targets = np.swapaxes(targets, 0, 1)

# Find best inputs.
inputs = SelectKBest(f_regression, k=n_best).fit_transform(inputs, targets.squeeze())

# Get names of best inputs.
print('\nThe best input features are:')
for i in range(inputs.shape[1]):
  field = inputs[:, i]
  name = get_name(field, day, idx_names)
  print(name)

# Split data randomly.
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.2)  

# Test.
model = fns.train(in_train, out_train)
pred, mse, mape, r2 = fns.test(model, in_test, out_test)
print('\nR2:', r2)
