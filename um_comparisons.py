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


def plot_diff_2d(data_1, data_2):
  # Unfinished. Need to allow for different dims and x-y positions. 
  # Avoid dividing by zero by setting 0 values to small values, ensuring that the original array isn't altered.
  data_1, data_2 = data_1.copy(), data_2.copy()
  data_1[data_1==0.0] = pow(10, -10)
  data_2[data_2==0.0] = pow(10, -10)

  diff = data_2 - data_1
  rel_diff = diff/data_1 * 100
  labels = ['Timesteps', 'Altitude / km', 'Latitude / degrees', 'Longitude / degrees']  
  x_label = labels[2]
  y_label = labels[1]
    
  fig, ((ax_1, ax_2), (ax_diff, ax_rel)) = plt.subplots(nrows=2, ncols=2, figsize=(12,9)) 
  ax_1.set_title('Original, averaged over timesteps and longitude')
  ax_2.set_title('Updated, averaged over timesteps and longitude')
  ax_diff.set_title('Difference') 
  ax_rel.set_title('Relative difference')  
  mesh_1 = ax_1.pcolormesh(data_1)
  mesh_2 = ax_2.pcolormesh(data_2)
  for mesh in [mesh_1, mesh_2]:
    bar = fig.colorbar(mesh)
    bar.set_label('J HCHO molecular / s\u207b\u00b9')
  mesh_diff = ax_diff.pcolormesh(diff)
  bar = fig.colorbar(mesh_diff)
  bar.set_label('New - old / s\u207b\u00b9')
  mesh_rel = ax_rel.pcolormesh(rel_diff)
  bar = fig.colorbar(mesh_rel)
  bar.set_label('Difference to original / %')  

  for ax in [ax_1, ax_2, ax_diff, ax_rel]:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
  fig.tight_layout(pad=4.0)
  plt.show() 
  del(data_1, data_2) # It should clean it up anyway but it never hurts to double check.


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
  
  
def pick_specifics(data):
  # Unfinished, but this is generally how it will work:
  data = data[0] # Pick one timestep.
  data = data[:,11,:,:] # Pick 1 altitude.
  #data = data[0,:,72,:] # Pick 1 latitude, the equator. 
  #data = data[0,:,:,60] # Pick 1 longitude.   
  return(data)
  
  
def zonal_means(data, dims):
  # dims is a list of True/False for which dims to keep.
  # False means average over it and remove the dim.
  j = 0 # Count how many dims were already removed.
  for i in range(len(dims)):
    if not dims[i]:
      data = np.mean(data, axis=i-j) # Average over dim.
      j += 1
  return(data)
  

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
dims = [False, True, True, False] # Time, alt, lat, long.

print(f'Loading the UM outputs with the {data_1_name}.')
data_1 = cf.read(file_1,select=code)[0] 
print(f'Loading the UM outputs with the {data_2_name}.')
data_2 = cf.read(file_2,select=code)[0] 

# One timestep. Cut off the top altitudes and the portion of space not receiving light.
#data_1 = data_1[0,0:60,0:108,0:160]
#data_2 = data_2[0,0:60,0:108,0:160]
# Need all times and longs for zonal means but a bit of space can be cut off to speed up.
data_1 = data_1[:,0:60,20:124,:]
data_2 = data_2[:,0:60,20:124,:]
print('Looking at this region:')
print(f'Time: {data_2.coord("time").data} UTC')
print(f'Altitude: {data_2.coord("atmosphere_hybrid_height_coordinate").data * 85000} km')
print('Latitude:', data_2.coord('latitude').data)
print('Longitude:', data_2.coord('longitude').data)

# This takes a long time but makes later operations easier.
npfile_1 = f'{path}numpy_data_original.npy'
npfile_2 = f'{path}numpy_data_updated.npy'
try:
  data_1 = np.load(npfile_1)
  data_1 = np.squeeze(data_1)
  assert(data_1.shape == data_2.shape)
  data_2 = np.load(npfile_2)
  data_2 = np.squeeze(data_2)
except:
  data_1 = np.array(data_1.data)
  data_2 = np.array(data_2.data)
  np.save(npfile_1, data_1)
  np.save(npfile_2, data_2)

# Avergae and reduce some dims.
data_1 = zonal_means(data_1, dims) 
data_2 = zonal_means(data_2, dims)

# View a 2D plot of the differences.   
plot_diff_2d(data_1, data_2) 
  
# See an overview of data and differences.  
flatfile_1 = f'{path}1d_data_original.npy'
flatfile_2 = f'{path}1d_data_updated.npy'
try:
  data_1 = np.load(flatfile_1)
  assert(data_1.size == data_2.size)
  data_2 = np.load(flatfile_2)   
except:
  data_1 = data_1.flatten() 
  data_2 = data_2.flatten()
  np.save(flatfile_1, data_1)
  np.save(flatfile_2, data_2)
  
view_values(data_1, data_1_name)
view_values(data_2, data_2_name)

max_1, min_1, avg_1 = view_basics(data_1, data_1_name)
max_2, min_2, avg_2 = view_basics(data_2, data_2_name)

print('\n')
simple_diff(max_1, max_2, data_1_name, data_2_name, 'largest', field_name)
simple_diff(min_1, min_2, data_1_name, data_2_name, 'smallest', field_name)
simple_diff(avg_1, avg_2, data_1_name, data_2_name, 'mean', field_name)

diffs(data_1, data_2, data_1_name, data_2_name)
print('\n')