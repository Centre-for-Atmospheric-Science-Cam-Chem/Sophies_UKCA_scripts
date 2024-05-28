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
show = fns.show
  
# Metadata functions.

def get_idx_names(name_file):
  # Names of the fields, taken from metadata.txt.
  # name_file: file path containing array of fields' indices and names.
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
    days = np.empty((84, 0))
    for i in range(len(input_files)):
      print(f'Adding day {i+1} to array.')
      start = time.time()
      npy_file = input_files[i]
      data = np.load(npy_file)
      days = np.append(days, data, axis=1)
      mem_used = psutil.virtual_memory().percent
      if mem_used > max_mem:
        sys.exit(f'The maximum memory usage limit of {max_mem}% was exceeded with a {i+1} day array of size {days.shape}.')
      end = time.time()
      seconds = round(end - start)
      print(f'That day took {seconds} seconds.')
      print(f'Size of array: {days.shape}.')
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
  

def shape(rows):
  # Make single input fields 2d so that they fit with other parameters going into the ML functions.
  # rows: np array of fields chosen for training or testing.
  if rows.ndim == 1:
    rows = np.vstack((rows, ))
  return(rows) 
