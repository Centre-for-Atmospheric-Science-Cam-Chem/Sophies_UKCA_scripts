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
import prediction_fns_shared as fns

train = fns.train
test = fns.test
force_axes = fns.force_axes
show = fns.show
  
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
    # Start timing the whole thing.
    start = time.time()    
    # For each grid & time point...
    n_points = len(data[0]) 
    print(f'This will take about {round(((n_points / 10000 * 6.5) + 1349) / 60 / 60)} hours.')      
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
    # See how long the whole thing took.
    end = time.time()
    mins = round((end - start) )
    print(f'The whole cloud column summing took {mins} minutes.')      
    # Save the new dataset.
    if out_path is not None:
      print('Saving new dataset.')
      start = time.time()
      np.save(out_path, data)
      end = time.time()
      mins = round((end - start) / 60)
      print(f'Saving the dataset took {mins} minutes.')
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
  cutoff = 20 # Pascals.
  bottom = data[:, np.where(data[7] > cutoff)].squeeze()
  top = data[:, np.where(data[7] < cutoff)].squeeze()
  return(bottom, top)
