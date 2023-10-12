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


def count_unique(plain_list):
  pd_list = pd.Series(plain_list)
  counts = pd_list.value_counts(ascending=False)
  return counts


# Investigate data values.
def view_values(data, name):
  num_values = len(data)
  counts = count_unique(data)
  print('\n')
  
  if max(counts) > 1:
    end = -3
    if len(counts) < 3:
      end = len(counts) * (-1)
    print(name, 'contains mostly:')
    print('value    count')
    print(counts.iloc[-1:end])

    print('\nvalue    % of data')
    print((counts.iloc[-1:end]/num_values)*100)
    print('\n')
  
  print(f'{name} largest value: {max(points)}')
  print('smallest value:', min(points))
  print('mean value:', np.nanmean(points))


def diffs(data_1, data_2, name_1, name_2):
  # data1 - data2 to see differences.
  # how many data are different?
  # what % of the data are different?
  # by how much do they differ?
  pass


# File paths.
path = '/scratch/st838/netscratch/jHCHO_update/'
file_1 = f'{path}hcho_orig_cristrat.pe19880901' # The original data.
file_2 = f'{path}hcho_updated_cristrat.pe19880901' # The updated data.

# Stash codes.
fields = [['HCHO radical','stash_code=50501'], # HCHO radical reaction.
          ['HCHO molecular','stash_code=50502']] # HCHO molecular reaction.

# What we want to compare.
field = fields[1]
field_name = field[0]
code = field[1]
print('Comparing', field_name)

data_1_name = 'old J data'
data_2_name = 'new J data'
print(f'Loading the UM outputs with the {data_1_name}.')
data_1 = cf.read(file_1,select=code)[0] 
print(f'Loading the UM outputs with the {data_2_name}.')
data_2 = cf.read(file_2,select=code)[0] 
