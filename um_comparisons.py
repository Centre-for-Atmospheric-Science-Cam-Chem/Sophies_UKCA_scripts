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
import time
import numpy as np

  
def view_values(data, name):
  # data: list or array of data points.
  start = time.monotonic()
  num_values = len(data)
  print('counting values')
  values, counts = np.unique(data, return_counts=True)
  print('checking counts')
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
        break
    
    print('\n')
  end = time.monotonic()
  print(f'view_values took {round((end-start), 1)} seconds.')
    

def view_basics_array(data, name):
  # data: list or np array.
  start = time.monotonic()
  largest = np.max(data)
  smallest = np.min(data)
  avg = np.mean(data)
  print(f'{name} largest value: {largest}')
  print('smallest value:', smallest)
  print('mean value:', avg)
  end = time.monotonic()
  print(f'view_basics_array took {round((end-start), 1)} seconds.')
  return(largest, smallest, avg)


def view_basics_field(data, name):
  # data: CF field. Takes longer than the array version.
  start = time.monotonic()
  largest = data.max()
  smallest = data.min()
  avg = data.mean()
  print(f'{name} largest value: {largest}')
  print('smallest value:', smallest)
  print('mean value:', avg)
  end = time.monotonic()
  print(f'view_basics_field took {round((end-start), 1)} seconds.')
  return(float(largest[0,0,0,0].data), float(smallest[0,0,0,0].data), float(avg[0,0,0,0].data))


def simple_diff(value_1, value_2, data_1_name, data_2_name, values_name, field_name):
  # Compare 2 numbers.
  # value_1 and value_2 are the nums to compare.
  # data_1_name and data_2_name are the names of the datasets.
  # values_name is what the values are e.g. 'largest', 'smallest', 'average'.
  # field_name describes what the stash item is.
  start = time.monotonic()
  if value_1 != value_2:
    diff = round(value_2 / value_1, 2)  
    print(f'The {values_name} {field_name} value from {data_2_name} is {diff} x the {values_name} value from {data_1_name}.')
  else:
    print(f'The {values_name} {field_name} values from both datasets are the same.')
  end = time.monotonic()
  print(f'simple_diff took {round((end-start), 1)} seconds.')


def plot_diffs():
  # Plot data2 - data1 by colour. 
  # Don't use flattened data?
  # Zonal means?
  pass


def diffs(data_1, data_2, name_1, name_2):
  # data1 - data2 to see differences.
  start = time.monotonic()
  num_values = len(data_1)
  diffs = data_2 - data_1
  # how many data are different?
  num_diff = np.count_nonzero(diffs)
  print(f'{num_diff} of {name_2} values are different to the corresponding {name_1} values.')
  # what % of the data are different?
  print(f'{(num_diff/num_values)*100}% of the data are different.')
  # by how much do they differ?
  view_basics_array(diffs, 'differences between the data')
  end = time.monotonic()
  print(f'simple_diff took {round((end-start), 1)} seconds.')


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
print('\nComparing', field_name)

data_1_name = 'old J data'
data_2_name = 'new J data'

both_data = [[data_1_name, file_1],
             [data_2_name, file_2]]

for dataset in both_data:
  print(f'Loading the UM outputs with the {dataset[0]}.')
  data = cf.read(dataset[1],select=code)[0] 
  data = data[0,0:60,0:108,0:160] # One timestep. Cut off the top altitudes and the portion of space not receiving light.
  #data = data[0,11,:,:] # Pick 1 altitude.
  data = data[0,:,72,:] # Pick 1 longitude. The equator.
  #data = data[0,:,:,60] # Pick 1 latitude.
  print('Looking at this region:')
  print(f'Time: {data.coord("time").data} UTC.')
  print('Altitude:', data.coord('atmosphere_hybrid_height_coordinate').data * 85000)
  print('Longitude:', data.coord('longitude').data)
  print('Latitude:', data.coord('latitude').data)
  print(f'Reducing dimensions.')
  start = time.monotonic()
  data = data.data.flatten()
  data = np.array(data)
  end = time.monotonic()
  print(f'faffing around with the data types took {round((end-start)/60, 1)} minutes.')
  view_values(data, dataset[0])
  max_value, min_value, avg_value = view_basics_array(data, dataset[0])
  del(data)
  dataset.append([max_value, min_value, avg_value])

print('\n')
simple_diff(both_data[0][2][0], both_data[1][2][0], data_1_name, data_2_name, 'largest', field_name)
simple_diff(both_data[0][2][1], both_data[1][2][1], data_1_name, data_2_name, 'smallest', field_name)
simple_diff(both_data[0][2][2], both_data[1][2][2], data_1_name, data_2_name, 'mean', field_name)

 
