'''
Name: Sophie Turner.
Date: 18/10/2023.
Contact: st838@cam.ac.uk
Simplified script to compare UKCA data and see differences for entire items. Will take a long time to run.
For a faster version with reduced dimensions see um_comparisons.py.
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
import cfplot as cfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

  
def view_values(data, name):
  # data: list or array of data points.
  num_values = len(data)
  values, counts = np.unique(data, return_counts=True)
  if any(val > 1 for val in counts):
    print(f'\n{name} contains mostly:')  
    print('value, count, % of data:')
    counts_temp = counts.copy()
    for _ in range(10):
      i = np.argmax(counts_temp)
      if counts_temp[i] > 1:
        print(values[i], counts_temp[i], (counts_temp[i]/num_values)*100)
        counts_temp[i] = 0
      else:
        print('All the other values occur only once.')
        break
      

def view_basics_array(data, name):
  # data: list or np array.
  largest = np.max(data)
  smallest = np.min(data)
  avg = np.mean(data)
  print(f'\nThe {name} range from {smallest} to {largest}.')
  print('mean value:', avg)
  return(largest, smallest, avg)


def view_basics_field(data, name):
  # data: CF field. Can take longer than the array version.
  largest = data.max()
  smallest = data.min()
  avg = data.mean()
  print(f'\nThe {name} range from {smallest} to {largest}.')
  print('mean value:', avg)
  return(float(largest[0,0,0,0].data), float(smallest[0,0,0,0].data), float(avg[0,0,0,0].data))


def simple_diff(value_1, value_2, data_1_name, data_2_name, values_name, field_name):
  # Compare 2 numbers.
  # value_1 and value_2 are the nums to compare.
  # data_1_name and data_2_name are the names of the datasets.
  # values_name is what the values are e.g. 'largest', 'smallest', 'average'.
  # field_name describes what the stash item is.
  if value_1 != value_2:
    diff = value_2/value_1  
    print(f'The {values_name} {field_name} value from {data_2_name} is {round(diff, 2)} x the {values_name} value from {data_1_name}.')
  else:
    print(f'The {values_name} {field_name} values from both datasets are the same.')


def diffs(data_1, data_2, name_1, name_2):
  # data1 - data2 to see differences.
  num_values = len(data_1)
  diffs = data_2 - data_1
  # how many data are different?
  num_diff = np.count_nonzero(diffs)
  print(f'\n{num_diff} of {name_2} values are different to the corresponding {name_1} values.')
  # what % of the data are different?
  print(f'{(num_diff/num_values)*100}% of the data are different.')
  # by how much do they differ?
  view_basics_array(diffs, 'differences')
  

# File paths.
path = '/scratch/st838/netscratch/jHCHO_update/ssp/'
file_1 = f'{path}all_original.pl20150102.pp' # The original data. Much larger file that file 2. Holds all J rates.
file_2 = f'{path}hcho_updated.pn20150102.pp' # The updated data. Contains 2 items.

# Stash codes.
fields = [['J HCHO radical','stash_code=50501'], # HCHO radical reaction.
          ['J HCHO molecular','stash_code=50502']] # HCHO molecular reaction.

# What we want to compare.
field = fields[1]
field_name = field[0]
code = field[1]
print('\nComparing', field_name)
data_1_name = 'old J data'
data_2_name = 'new J data'

print(f'Loading the UM outputs with the {data_1_name}.')
data_1 = cf.read(file_1,select=code)[0] 
print(f'Loading the UM outputs with the {data_2_name}.')
data_2 = cf.read(file_2,select=code)[0] 

data_1 = np.array(data_1.data)
data_2 = np.array(data_2.data)

print('Looking at this region:')
print(f'Time: {data_2.coord("time").data} UTC')
print(f'Altitude: {data_2.coord("atmosphere_hybrid_height_coordinate").data * 85000} km')
print('Longitude:', data_2.coord('longitude').data)
print('Latitude:', data_2.coord('latitude').data)
    
# See an overview of data and differences.    
data_1 = data_1.flatten() 
data_2 = data_2.flatten()
view_values(data_1, data_1_name)
view_values(data_2, data_2_name)
max_1, min_1, avg_1 = view_basics_array(data_1, data_1_name)
max_2, min_2, avg_2 = view_basics_array(data_2, data_2_name)

print('\n')
simple_diff(max_1, max_2, data_1_name, data_2_name, 'largest', field_name)
simple_diff(min_1, min_2, data_1_name, data_2_name, 'smallest', field_name)
simple_diff(avg_1, avg_2, data_1_name, data_2_name, 'mean', field_name)

diffs(data_1, data_2, data_1_name, data_2_name)
print('\n')



