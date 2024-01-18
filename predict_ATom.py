'''
Name: Sophie Turner.
Date: 16/1/2023.
Contact: st838@cam.ac.uk
Try to predict J rates using ML with the ATom data.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime


def standard_names(data):
  # Standardise the names of ATom fields into a simple format.
  for name_old in data.iloc[:,4:-1].columns:
    name_new = name_old.replace('CAFS', '')
    name_new = name_new.replace('_', ' ')
    name_new = name_new.upper()  
    name_new = name_new.strip()
    data = data.rename(columns={name_old:name_new})
  return(data)
  

def normalise(col):
  # Col: a dataframe column.
  return(col - col.min()) / (col.max() - col.min())  
  
  
def prep_params(target_name, input_name, ATom_data, J_all):
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
  ATom_data = ATom_data.dropna(axis=0, subset=all_used) 

  # Normalise relevant data.
  #for col in all_used:
  #  ATom_data[col] = normalise(ATom_data[col])

  targets = ATom_data[target_name]
  inputs = ATom_data[input_name] 
  
  # Split data (almost randomly).
  in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.2, random_state=6)
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
  mse = mean_squared_error(out_test, out_pred)
  mse = round(mse, 2)
  r2 = r2_score(out_test, out_pred)
  r2 = round(r2, 2)
  return(out_pred, mse, r2)
  
  
def show(out_test, out_pred, mse, r2):
  print(f'Mean squared error = {mse}')
  print(f'Coefficient of determination = {r2}')
  plt.scatter(out_test, out_pred)
  plt.xlabel('actual')
  plt.ylabel('predicted')
  plt.show()  
  

# File paths.
dir_path = '/scratch/st838/netscratch/'
ATom_dir = dir_path + 'ATom_MER10_Dataset/'
ATom_file = ATom_dir + 'photolysis_data.csv'

# Open ATom dataset, already partially pre-processed.
ATom_data = pd.read_csv(ATom_file)
ATom_data = ATom_data.rename(columns={'UTC_Start_dt':'TIME', 'T':'TEMPERATURE', 'G_LAT':'LATITUDE', 
                                      'G_LONG':'LONGITUDE', 'G_ALT':'ALTITUDE', 'Pres':'PRESSURE',
				      'cloudindicator_CAPS':'CLOUD %'})    
ATom_data = standard_names(ATom_data)  

# Convert time info into datetime and float types.
# Formatted as str like '2017-10-04 22:55:00'.
seconds = []
for row in ATom_data.index: 
  time_str = ATom_data.loc[row]['TIME']
  time_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
  second = time_dt.timestamp()
  seconds.append(second)
ATom_data.insert(1, 'SECONDS', seconds)

# Set up data structures for different experiments.
not_used = ['TIME', 'CLOUDFLAG AMS', 'JO3 DNWFRAC', 'JNO2 DNWFRAC']
ATom_data = ATom_data.drop(columns=not_used)
J_all = ATom_data.loc[:,ATom_data.columns.str.startswith('J')].columns.tolist()
phys_all = ['SECONDS', 'TEMPERATURE', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 
            'PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY']			  
phys_reduced = ['TEMPERATURE', 'PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY']
phys_min = ['PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE']

# Test using each J as the only input.
best, worst = [], []
for J in J_all:
  input_name = [J]
  target_name = J_all.copy()
  in_train, in_test, out_train, out_test = prep_params(target_name, input_name, ATom_data, J_all)
  model = train(in_train, out_train)
  out_pred, mse, r2 = test(model, in_test, out_test)
  if r2 >= 0.88:
    best.append([input_name[0], r2])
    show(out_test, out_pred, mse, r2)
  elif r2 <= 0.6:
    worst.append([input_name[0], r2])

# Look at the results.
best.sort(reverse = True, key = lambda x: x[1])
worst.sort(key = lambda x: x[1])
print('\nbest:')
for J in best:
  print(J)
print('worst:')
for J in worst:
  print(J)   
# The best from the above were JH2O2 OH OH, JCH3OOH CH3O OH and JCHOCHO CH2O CO with r2 0.89.

input_name = phys_all
target_name = ['JNO2 NO O3P']
in_train, in_test, out_train, out_test = prep_params(target_name, input_name, ATom_data, J_all)
model = train(in_train, out_train)
out_pred, mse, r2 = test(model, in_test, out_test)
show(out_test, out_pred, mse, r2)
