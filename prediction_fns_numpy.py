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
import matplotlib.pyplot as plt
import prediction_fns_shared as fns

train = fns.train
test = fns.test
force_axes = fns.force_axes
show = fns.show
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
    
def only_range(inputs, targets, targets_all, bottom=0, top=1, min_samples=10):
  # Only use a certain range of values of J rates.
  # Call just before fns.test() and pass only test sets if you want model trained on all data but only tested on smallest.
  # inputs: array of inputs (in_test for test-only).
  # targets: array of targets (out_test for test-only).  
  # targets_all: array of all targets in data, used for the test-only option. If using all data, just pass the targets for both target params.
  # bottom: chosen proportion of data for lowest value, 0 to 1, inclusive.
  # top: chosen proportion of data for highest value, 0 to 1, inclusive.
  # min_samples: minimum number of samples a J rate needs for it to be included. 
  assert bottom < top
  if targets.ndim == 1:
    top_all = max(targets_all)
    bottom_new = top_all * bottom
    top_new = top_all * top
    i_range = np.where((targets >= bottom_new) & (targets <= top_new))
    # Reduce the data to these indices.
    inputs = inputs[i_range]
    targets = targets[i_range]
  else:
    deletes = np.array([], dtype=int)
    # For each target feature...
    for i in range(len(targets_all[0])):
      target = targets_all[:, i]
      # Get the index.
      top_all = max(target)
      bottom_new = top_all * bottom
      top_new = top_all * top
      i_range = np.where((targets[:, i] >= bottom_new) & (targets[:, i] <= top_new)) 
      # Sometimes there are no data in this region for some J rates.
      if len(i_range[0]) < min_samples:
        deletes = np.append(deletes, i)
      else:
        # Reduce the data to these indices.
        inputs = inputs[i_range]
        targets = targets[i_range]
    # Remove these J rates from the selection.
    targets = np.delete(targets, deletes, axis=1)
  return(inputs, targets, deletes)  
  
  
# Results functions.
  
def show_col_orig(data, ij=78, name='O3', all_time=True):
  # Show a column of J rate by altitude for full UKCA datasets before ML. 
  # data: 2d np array of full dataset.
  # ij: index of J rate to look at. 78=O3 in full dataset.
  # name: name of J rate.
  # all_time: whether to include all time steps (True) or plot a line of one time point (False).
  # Pick a lat-lon point and a time.
  # Cambridge at midday.
  lat, lon, hour = 51.875, 0.9375, 12
  #lat, lon, hour = 89.375, 0.9375, 12 # The north pole at midday.
  #lat, lon, hour = -89.375, 0.9375, 12 # The south pole at midday. 
  #lat, lon, hour = 0.625, 0.9375, 12 # Gulf of Guinea at midday.
  # Choose whether to specify the time point.
  if not all_time:
    # Find the first occurance of this time so that we only select one day.
    data = data[:, np.where(data[0] == hour)].squeeze()
    dt0 = np.unique(data[4])[0]
    data = data[:, np.where(data[4] == dt0)].squeeze()
  # Fetch this grid col's data.
  data = data[:, np.where((data[2] == lat) & (data[3] == lon))].squeeze() 
  # Pick out the J rate and altitude.
  j = data[ij]
  alts = data[1]
  alts = alts * 85
  # Plot the J rate by altitude in that column.
  if all_time:
    plt.scatter(j, alts, label=f'J{name} from UKCA', alpha=0.2)
    plt.title('UKCA column ozone J rates over Cambridge in 2015')
  else:
    plt.plot(j, alts, label=f'J{name} from UKCA')
    plt.title('UKCA column ozone J rates over Cambridge at midday on 15/1/2015')
  plt.xlabel('O3 -> O2 + O(1D) J rate')
  plt.ylabel('Altitude / km')
  plt.show()
  plt.close()
 
 
def col(data, coords, lat, lon, hour, ij):
  # Show a column of J rate by altitude to compare targets and predictions. 
  # data: 2d np array dataset e.g. full UKCA data or ML targets.
  # lat, lon: latitude and longitude of chosen column.
  # hour: hour of day of chosen sample or None if plotting all time points.
  # ij: index of J rate to look at.
  # Stick the coords on for easier processing.
  if data.ndim == 1:
    data = np.expand_dims(data, 1)
  data = np.append(coords, data, axis=1)
  if hour is not None:
    # Find the first occurance of this time so that we only select one day.
    data = data[np.where(data[:, 0] == hour)].squeeze()
    dt0 = np.unique(data[:, 4])[0]
    data = data[np.where(data[:, 4] == dt0)].squeeze()
  # Fetch this grid col's data.
  data = data[np.where((data[:, 2] == lat) & (data[:, 3] == lon))].squeeze() 
  # Pick out the J rate and altitude.
  ij += 5 # Account for added coords.
  j = data[:, ij]
  alts = data[:, 1]
  alts = alts * 85 # Convert proportion to km.
  return(j, alts)
  
  
def show_col(out_test, out_pred, coords, ij, name='O3', all_time=True):
  # Show a column of J rate by altitude. 
  # out_test: 2d np array dataset of targets.
  # out_pred: 2d np array dataset of predictions.
  # coords: 2d np array of time & space co-ordinates for datasets.
  # ij: index of J rate to look at.
  # name: name of J rate.
  # all_time: whether to include all time steps (True) or plot a line of one time point (False).
  # Pick a lat-lon point and a time.
  lat, lon, hour = 51.875, 0.9375, 12 # Cambridge at midday.
  #lat, lon, hour = 89.375, 0.9375, 12 # The north pole at midday.
  #lat, lon, hour = -89.375, 0.9375, 12 # The south pole at midday. 
  #lat, lon, hour = 0.625, 0.9375, 12 # Gulf of Guinea at midday.
  # Swap the dimensions to make them compatible with sklearn.
  coords = np.swapaxes(coords, 0, 1)
  # Choose whether to specify the time point.
  if all_time:
    hour = None
  j1, alts = col(out_test, coords, lat, lon, hour, ij)
  j2, alts = col(out_pred, coords, lat, lon, hour, ij)  
  # Plot the J rate by altitude in that column.
  if all_time:
    plt.scatter(j1, alts, label=f'J{name} from UKCA', marker='|', s=50, alpha=0.5)
    plt.scatter(j2, alts, label=f'J{name} from random forest', marker='_', s=50, alpha=0.5)
    plt.title('UKCA J rates, column over Cambridge in 2015')
  else:
    plt.plot(j1, alts, label=f'J{name} from UKCA')
    plt.plot(j2, alts, label=f'J{name} from random forest') 
    plt.title(f'UKCA J{name}, column over Cambridge at midday on 15/7/2015')
  plt.legend()    
  plt.xlabel(f'J{name} / s\u207b\u00b9')
  plt.ylabel('Altitude / km')
  plt.show()
  # Column percentage difference plot.
  diff = np.nan_to_num(((j2 - j1) / j1) * 100, posinf=0, neginf=0) # Accounts for div by zero.
  if all_time:
    plt.hist(diff, bins=100)
  else:
    plt.plot(diff, alts)
    plt.axvline(linestyle=':')
  plt.title(f'Column % difference of J{name} predictions to UKCA outputs')
  plt.xlabel('% difference')
  plt.ylabel('Altitude / km')
  plt.show()
  plt.close() 
  
