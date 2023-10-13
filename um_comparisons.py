'''
Name: Sophie Turner.
Date: 12/10/2023.
Contact: st838@cam.ac.uk
Script to compare UKCA data and see differences.
Used on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

# Stop unnecessary warnings about pandas version.
import warnings
warnings.simplefilter('ignore')

import cf
import numpy as np

  
def view_values(data, name):
  # I think the memory is filling up or overflowing.
  # Not using this fn for now.
  num_values = len(data)
  print('counting values')
  values, counts = np.unique(data, return_counts=True)
  print('checking counts')
  if any(val > 1 for val in counts):
    print(name, 'contains mostly:')
    print('value    count')
    counts_temp = counts.copy()
    for _ in range(3):
      i = np.argmax(counts_temp)
      print(values[i], counts_temp[i])
      counts_temp[i] = 0

    print('\nvalue    % of data')
    counts_temp = counts.copy()
    for _ in range(3):
      i = np.argmax(counts_temp)
      print(values[i], (counts_temp[i]/num_values)*100)
      counts_temp[i] = 0
    
    print('\n')
    

def view_basics(data, name):
  largest = max(data)
  smallest = min(data)
  avg = np.mean(data)
  print(f'{name} largest value: {largest}')
  print('smallest value:', smallest)
  print('mean value:', avg)
  return(largest, smallest, avg)


def simple_diff(value_1, value_2, data_1_name, data_2_name, values_name, field_name):
  # Compare 2 numbers.
  # value_1 and value_2 are the nums to compare.
  # data_1_name and data_2_name are the names of the datasets.
  # values_name is what the values are e.g. 'largest', 'smallest', 'average'.
  # field_name describes what the stash item is.
  if value_1 != value_2:
    diff = round(value_2 / value_1, 2)  
    print(f'The {values_name} {field_name} value from {data_2_name} is {diff}x the {values_name} value from {data_1_name}.')
  else:
    print(f'The {values_name} {field_name} values from both datasets are the same.')


def plot_diffs():
  # Plot data2 - data1 by colour. 
  # Don't use flattened data?
  # Zonal means?
  pass


def diffs(data_1, data_2, name_1, name_2):
  # data1 - data2 to see differences.
  # how many data are different?
  # what % of the data are different?
  # by how much do they differ?
  pass


# File paths.
path = '/scratch/st838/netscratch/jHCHO_update/ssp/'
file_1 = f'{path}all_original.pl20150102.pp' # The original data. Much larger file that file 2. Holds all J rates.
file_2 = f'{path}hcho_updated.pn20150102.pp' # The updated data. Contains 2 items.

# Stash codes.
fields = [['J-HCHO radical','stash_code=50501'], # HCHO radical reaction.
          ['J-HCHO molecular','stash_code=50502']] # HCHO molecular reaction.

# What we want to compare.
field = fields[1]
field_name = field[0]
code = field[1]
print('Comparing', field_name)

data_1_name = 'old J data'
data_2_name = 'new J data'

print(f'Loading the UM outputs with the {data_1_name}.')
data_1 = cf.read(file_1,select=code)[0] 
data_1 = data_1[12]
print(f'Loading the UM outputs with the {data_2_name}.')
data_2 = cf.read(file_2,select=code)[0] 
data_2 = data_2[12]

print(f'Reducing dimensions.')
data_1_1d = data_1.data.flatten()
data_2 = data_2.data.flatten() 

max_1, min_1, avg_1 = view_basics(data_1_1d, data_1_name)
max_2, min_2, avg_2 = view_basics(data_2, data_2_name)
print('\n')
simple_diff(max_1, max_2, data_1_name, data_2_name, 'largest', field_name)
simple_diff(min_1, min_2, data_1_name, data_2_name, 'smallest', field_name)
simple_diff(avg_1, avg_2, data_1_name, data_2_name, 'mean', field_name)
