'''
Name: Sophie Turner.
Date: 19/1/2023.
Contact: st838@cam.ac.uk
Functions which are used by prediction scripts.
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def standard_names(data):
  # Standardise the names of ATom fields into a simple format.
  for name_old in data.iloc[:,4:-1].columns:
    name_new = name_old.replace('CAFS', '')
    name_new = name_new.replace('_', ' ')
    name_new = name_new.upper()  
    name_new = name_new.strip()
    data = data.rename(columns={name_old:name_new})
  return(data)
  
  
def time_to_s(data):
  # Adds a new column to dataset with time info in float of seconds.
  # TIME col is formatted as str like '2017-10-04 22:55:00'.
  seconds = []
  for row in data.index: 
    time_str = data.loc[row]['TIME']
    time_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    second = time_dt.timestamp()
    seconds.append(second)
  data.insert(1, 'SECONDS', seconds)
  return(data)
  

def remove_tiny(col_name, data1, data2=None, percentage=1):
  # Remove J rate data in the smallest 1% of values (not the smallest 1% of data).
  # col_name: name of a column of J rate values.
  # data1: dataframe containing col.
  # data2: optional 2nd dataset for consistency.
  # percent: how many % to take off the bottom.
  # Avoid using on lots of cols as it might remove too much data. 
  col_name = col_name[0]
  col = data1[col_name]
  span = col.max() - col.min()
  percent = span / 100
  percent = percent * percentage
  lowest = col.min() + percent
  if data2 is not None:
    data2 = data2.drop(data1[data1[col_name] < lowest].index)
  data1 = data1.drop(data1[data1[col_name] < lowest].index) 
  return(data1, data2)


def normalise(col):
  # col: a dataframe column.
  return(col - col.min()) / (col.max() - col.min())  
  
  
def prep_data(target_name, input_name, data, J_all):
  # Prepare dataset, when inputs and targets come from the same pandas dataset.
  # Don't let targets into inputs.
  if target_name == J_all:
    for name in input_name:
      if name[0] == 'J':
        target_name.remove(name) 
  elif input_name == J_all:
    for name in target_name:
      if name[0] == 'J':
        input_name.remove(name)

  # Remove relevant rows with empty fields.
  # Removes up to 92% of the data!
  all_used = input_name + target_name
  data = data.dropna(axis=0, subset=all_used) 

  # Normalise relevant data.
  #for col in all_used:
  #  data[col] = normalise(data[col])

  return(target_name, input_name, data)


def set_params(target_name, input_name, target_data, input_data):
  # For pandas datasets.
  targets = target_data[target_name]
  inputs = input_data[input_name] 
  # Split data (almost randomly).
  in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1, random_state=6)
  return(in_train, in_test, out_train, out_test)
  
  
def train(in_train, out_train):
  # Set up simple model.
  model = linear_model.LinearRegression()
  model.fit(in_train, out_train)
  return(model)


def test(model, in_test, out_test):
  # Try it out.
  out_pred = model.predict(in_test)
  # See how well it did.
  err = mean_absolute_error(out_test, out_pred)
  #err = round(err, 2)
  r2 = r2_score(out_test, out_pred)
  r2 = round(r2, 2)
  return(out_pred, err, r2)
  
  
def show(out_test, out_pred, mae, r2):
  print(f'Mean squared error = {mae}')
  print(f'Coefficient of determination = {r2}')
  plt.scatter(out_test, out_pred)
  plt.xlabel('actual')
  plt.ylabel('predicted')
  plt.show() 
