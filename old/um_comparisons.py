'''
Name: Sophie Turner.
Date: 30/10/2023.
Contact: st838@cam.ac.uk
Script to compare pre-procecced ATom and UKCA data and see differences.
Used on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cartopy.crs as ccrs


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
      

def view_basics(data, name):
  # data: list or np array.
  largest = np.max(data)
  smallest = np.min(data)
  avg = np.mean(data)
  print(f'\nThe {name} range from {smallest} to {largest}.')
  print(f'Mean value of {name}:', avg)
  return(largest, smallest, avg)


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
  diff = data_2 - data_1
  rel_diff = diff/data_1 * 100
  # how many data are different?
  num_diff = np.count_nonzero(diff)
  print(f'\n{num_diff} of {name_2} values are different to the corresponding {name_1} values.')
  # what % of the data are different?
  print(f'{(num_diff/num_values)*100}% of the data are different.')
  # by how much do they differ?
  view_basics(diff, 'differences')
  view_basics(rel_diff, 'relative differences %')
  
  
def plot_both(data_1, data_2):
  plt.figure()
  plt.title = data_1.name
  x = data_1.index
  y1 = data_1
  y2 = data_2
  plt.plot(x, y1)
  plt.plot(x, y2)
  plt.show()
  

def plot_diff(data_1, data_2):
  plt.figure()
  plt.title = data_1.name
  diff = data_2 - data_1
  rel_diff = diff/data * 100
  plt.hist(rel_diff)
  plt.show()
  
  
def plot_location(data):
  date = data.index[0]
  date = date.split()[0]
  fig = pylab.figure(dpi=150)
  ax = pylab.axes(projection=ccrs.PlateCarree(central_longitude=310.0))
  ax.stock_img()
  pylab.title(f'Locations of selected flight path points, {date}')
  x = data['LONGITUDE']
  y = data['LATITUDE']
  pylab.scatter(x+50, y, s=7, c='crimson')
  pylab.show()
  
 
# File paths.
path = '/scratch/st838/netscratch/tests/'
ATom_file = f'{path}ATom_points_matched.csv'
UKCA_file = f'{path}UKCA_points_matched.csv'
 
ATom_data = pd.read_csv(ATom_file, index_col=0)
UKCA_data = pd.read_csv(UKCA_file, index_col=0) 
 
# Turn this into saves in a specific dir so it doesn't show pop ups. 
plot_location(ATom_data)
for field in ATom_data.columns:
  plot_both(ATom_data[field], UKCA_data[field])
  plot_diff(ATom_data[field], UKCA_data[field])



