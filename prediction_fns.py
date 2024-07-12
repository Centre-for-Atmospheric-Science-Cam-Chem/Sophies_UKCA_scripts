'''
Name: Sophie Turner.
Date: 21/6/2024.
Contact: st838@cam.ac.uk
Functions which are used by prediction scripts.
'''

import os
import sys
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

  
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
    days = np.empty((85, 0), dtype=np.float32)
    for i in range(len(input_files)):
      print(f'\nAdding day {i+1} to array.')
      start = time.time()
      npy_file = input_files[i]
      data = np.load(npy_file)
      days = np.append(days, data, axis=1)
      mem_used = psutil.virtual_memory().percent
      if mem_used > max_mem:
        sys.exit(f'The maximum memory usage limit of {max_mem}% was exceeded with a {i+1} day array of size {days.shape}.')
      end = time.time()
      seconds = round(end - start)
      gb = round(psutil.virtual_memory().used / 1000000000, 3) 
      print(f'That day took {seconds} seconds.')
      print(f'Size of array: {days.shape}.')            
      print(f'Memory usage: {gb} GB.')
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
  

# ML functions.  
  
def train(in_train, out_train):
  # Set up simple model.
  model = linear_model.LinearRegression()
  model.fit(in_train, out_train)
  return(model)


def test(model, in_test, out_test):
  # Try it out.
  out_pred = model.predict(in_test)
  # See how well it did.
  mse = mean_squared_error(out_test, out_pred)
  mape = mean_absolute_percentage_error(out_test, out_pred)
  r2 = r2_score(out_test, out_pred)
  return(out_pred, mse, mape, r2)  
  
  
# Results functions.  
  
def show(out_test, out_pred, title='', path=None):
  # Plot results of ML. 
  # out_test: np array of known test targets.
  # out_pred: np array of ML test outputs.
  # title: optional string for plot title.
  # path: optional file path string for saving plot.
  # Make sure they're the right shape.
  out_pred = out_pred.squeeze()
  out_test = out_test.squeeze()
  
  length = len(out_pred)
  if length > 10000:
    # Plotting this many datapoints is excessive and costly. Reduce it to 1%.
    idxs = np.arange(0, length, 100)
    out_pred = out_pred[idxs]
    out_test = out_test[idxs]
    del(idxs)

  plt.figure()
  plt.scatter(out_test, out_pred)
  plt.title(title)
  plt.xlabel('UKCA J rate / s\u207b\u00b9')
  plt.ylabel('Predicted J rate / s\u207b\u00b9')
  if path is None:
    plt.show()
  else:
    plt.savefig(path)
  plt.close()
