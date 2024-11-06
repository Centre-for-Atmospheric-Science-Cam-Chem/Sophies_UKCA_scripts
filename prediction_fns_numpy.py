'''
Name: Sophie Turner.
Date: 28/5/2024.
Contact: st838@cam.ac.uk
Functions which are used on numpy data by prediction scripts.
Files are located at scratch/st838/netscratch.
'''

import os
import sys
import time
import psutil
import numpy as np
import constants as con
import prediction_fns_shared as fns

train = fns.train
test = fns.test
force_axes = fns.force_axes
show = fns.show
show_col_orig = fns.show_col_orig
col = fns.col
show_col = fns.show_col
shrink = fns.shrink
force_axes = fns.force_axes
mem = fns.mem
  
# Metadata functions.

def get_idx_names(name_file):
  # Names of the fields, taken from metadata.
  # name_file: file path containing array of fields' indices and names.
  # Returns a list of indices and names.
  lines = open(name_file, 'r')
  idx_names = []
  for line in lines: 
    words = line.replace('\n', '')
    words = words.split(maxsplit=1) 
    lst = [words[0], words[1]]
    idx_names.append(lst)
  return(idx_names)  
      
  
# Data preprocessing functions.  
  
def collate(training_data_path, input_files):
  # Collate the data from multiple files, if it hasn't already been done.
  # training_data_path: file path of where to save the combined data.
  # input_files: list of training data file paths.
  if os.path.exists(training_data_path):
    days = np.load(training_data_path)
  else:
    # Memory usage limit.
    max_mem = 75 # Percentage.
    days = np.empty((85, 0), dtype=np.float32)
    for i in range(len(input_files)):
      print(f'Adding day {i+1} to array.')
      start = time.time()
      npy_file = input_files[i]
      data = np.load(npy_file)
      days = np.append(days, data, axis=1)
      mem_used = round(psutil.virtual_memory().percent, 1)
      if mem_used > max_mem:
        sys.exit(f'The maximum memory usage limit of {max_mem}% was exceeded with a {i+1} day array of size {days.shape}.')
      end = time.time()
      seconds = round(end - start)
      gb = round(psutil.virtual_memory().used / 1000000000, 3)
      print(f'That day took {seconds} seconds.')
      print(f'Size of array: {days.shape}.')
      print(f'Memory used so far: {gb} GB.')
      print(f'Memory usage at {mem_used}%.')
    # Save the np array for training data for re-use.
    # The .npy should be deleted after final use to free up space as it is very large.
    print('Saving .npy.')
    start = time.time()
    np.save(training_data_path, days)
    end = time.time()
    seconds = round(end - start)
    print(f'That took {seconds} seconds.')
  # Return one np array containing data from multiple files.
  return(days)


def find_cloud_end(data):
  # Find out at what level there are no more clouds.
  # data: np array of the contents of an entire .npy file of data, of shape (features, samples).     
  # For each grid & time point...
  n_points = len(data[0])
  # First altitude value.
  first_alt = data[1,0] 
  # Stride = lats x lons to check each altitude.
  stride = 144 * 192 
  j = 0
  for i in range(0, n_points, stride):      
    j += 1
    alt = data[1,i]
    # The altitudes are repeated in the data structure due to flattening. Don't process same ones again.
    if i != 0 and alt == first_alt:
      break
    # Indices of all data points at this altitude.  
    layer = np.where(data[1] == alt)  
    # Total cloud in this layer.
    cloud_in_layer = np.sum(data[6, layer])
    if cloud_in_layer == 0:
      break
  return(alt)    

  
def sum_cloud(data, out_path=None):
  # Sum the cloud in the column above each grid box and include that as a feature in training data.
  # data: np array of the contents of an entire .npy file of data, of shape (features, samples).  
  # See if the summed cloud col already exists.
  if len(data) < 86: 
    print('Summing cloud columns.')
    # Check at which altitude clouds stop.
    clear = find_cloud_end(data)
    # Make a new array for cloud col.
    col = np.zeros(data.shape[1], dtype=np.float32)   
    # For each grid & time point...
    n_points = len(data[0]) 
    print(f'This will take a while. At least half an hour. Go for a walk or something.')      
    for i in range(n_points):   
      # Get the time, altitude, latitude and longitude.
      alt = data[1,i]
      # Ignore any altitudes which are too high for clouds.
      if alt >= clear:
        continue
      lat = data[2,i]
      lon = data[3,i]
      dt = data[4,i] # date-time
      cloud = data[6,i]
      # Search only grid boxes after this one, not before.
      # Stride = lats x lons.
      stride = 144 * 192      
      for j in range(i, n_points, stride):
        alt_j = data[1,j]
	# Ignore any altitudes which are too high for clouds.
        if alt_j >= clear:
          break
        lat_j = data[2,j]
        lon_j = data[3,j]
        dt_j = data[4,j] # date-time
	# Get all the grid boxes where time, lat & lon are the same AND alt > this alt.
        if dt_j == dt and alt_j > alt and lat_j == lat and lon_j == lon:
          # Sum the cloud of those grid boxes.
          cloud_j = data[6,j]
          cloud += cloud_j
      # Add this number to the new feature at this grid box.
      col[i] = cloud
    # Add the new array as a feature in the dataset.
    data = np.insert(data, 7, col, axis=0)       
    # Save the new dataset.
    if out_path is not None:
      print('Saving new dataset.')
      np.save(out_path, data)
  # Return the new dataset.
  return(data)
  

def shape(rows):
  # Make single input fields 2d so that they fit with other parameters going into the ML functions.
  # rows: np array of fields chosen for training or testing.
  if rows.ndim == 1:
    rows = np.vstack((rows, ))
  return(rows) 
  
  
def remove_all_zero(data):
  # Remove all-zero fields from dataset.
  removes = []
  for i in range(len(data)):
    field = data[i]
    if np.all(field == 0):
      removes.append(i)
  data = np.delete(data, removes, axis=0)
  return(data)
  
  
def all_zeros(field):
  # Check if a field contains only zeros.
  zeros = False
  if np.all(field == 0):
    zeros = True
  return(zeros)
  
 
def get_name(idx, idx_names):
  # Get the name of a field from its index number.
  for idx_name in idx_names:
    if idx_name[0] == idx:
      return(idx_name[1])
      
  
def split_pressure(data):
  # Split the UKCA dataset into two: one above and one below the Fast-J cutoff, using pressure.
  # Data: 2D np array of whole UKCA dataset.
  cutoff = 20 # Pascals.
  # Check if the dataset includes cloud col and choose pressure index accordingly.
  if len(data) > 85:
    iP = 8
  else:
    iP = 7  
  bottom = data[:, np.where(data[iP] > cutoff)].squeeze()
  top = data[:, np.where(data[iP] < cutoff)].squeeze()
  return(bottom, top)
  
  
def sample(data, size=None):
  # Make a smaller dataset by randomly sampling the data, uniformly.
  # Choose a big enough size to capture sufficient tropospheric density.
  # To do: use a better function than random uniform sampling.
  # data: np array of dataset, 1D or 2D of shape (features, samples).
  # size: number of data points desired. Leave empty for auto.
  if data.ndim == 1:
    length = len(data)
  else: 
    length = len(data[0])
  if size is None:
    size = round(length / 10) # 10%.
  i = con.rng.integers(0, length, size, dtype=np.int32)
  if data.ndim == 1:
    data = data[i] 
  else:
    data = data[:, i] 
  return(data)
  
  
def tts(data, i_inputs, i_targets, test_size=0.1):
  # Train test split function which returns indices for coords.
  # data: 2d np array of full dataset containing all inputs and targets.
  # i_inputs: array of indices of input fields.
  # i_targets: array of indices of target fields.
  # test_size: float out of 1, proportion of data to use for test set. 
  # Find length of data and size of test set.
  len_all = len(data[0])
  len_test = round(len_all * test_size)
  i_all = np.arange(0, len_all, dtype=int)
  # Generate random, unique indices for selecting test set.
  i_test = con.rng.choice(i_all, len_test, replace=False) 
  i_test = np.sort(i_test)
  test_data = data[:, i_test]
  # Remove these from the training set.
  train_data = np.delete(data, i_test, axis=1)  
  # Get inputs and targets.
  in_train = train_data[i_inputs] 
  in_test = test_data[i_inputs]
  out_train = train_data[i_targets]
  out_test = test_data[i_targets]
  # Swap the dimensions to make them compatible with sklearn.
  in_train = np.swapaxes(in_train, 0, 1)
  in_test = np.swapaxes(in_test, 0, 1)
  out_train = np.swapaxes(out_train, 0, 1)
  out_test = np.swapaxes(out_test, 0, 1)
  # Return train and test datasets and their indices from the full dataset.
  return(in_train, in_test, out_train, out_test, i_test)
 
  
# ML functions.

def only_small(targets, in_test, out_test):
  # Only test ML model on smallest 10% of values of J rates (not smallest 10% of data).
  # Call just before fns.test() to ensure model is trained on all data but only tested on smallest.
  # targets: array of targets.
  # in_test: array of test inputs.
  # out_test: array of test targets.   
  if out_test.ndim == 1:
    top = max(targets)
    bottom = top / 10
    i_smallest = np.where(out_test <= bottom)
    in_test = in_test[i_smallest]
    out_test = out_test[i_smallest]
  else:
    # For each target feature...
    for i in range(len(targets[0])):
      target = targets[:, i]
      # Get the index of the smallest 10%
      top = max(target)
      bottom = top / 10
      i_smallest = np.where(out_test[:, i] <= bottom)
      # Reduce all the data to these indices.
      in_test = in_test[i_smallest]
      out_test = out_test[i_smallest]  
  return(in_test, out_test)
