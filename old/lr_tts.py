'''
Name: Sophie Turner.
Date: 11/8/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with Lasso regression using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''

import glob
import numpy as np
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn import linear_model
import prediction_fns_numpy as fns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_file = f'{paths.npy}/4days.npy'
#data_file = f'{paths.npy}/20150715.npy'

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
# Best physics inputs from feature selection.
phys_best = [1,7,8,9,10,14]
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
NO3 = 25
H2SO4 = 29
HOCl = 71
H2O2 = 74
O3 = 78 # O(1D) product.
# J rates which are not summed or duplicate fg.
J_core = [16,18,19,20,24,25,28,29,30,31,32,33,34,36,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82]
# The above with usually zero rates removed.
J_core = [16,18,19,20,24,25,28,30,31,32,33,51,52,66,68,70,71,72,73,74,75,76,78,79,80,81,82]
# Some favourites for testing.
J_test = [NO2,HCHOm,H2O2,O3]
J_varied = [NO2,HCHOm,NO3,H2SO4,HOCl,H2O2,O3]

data = np.load(data_file)

# Split the dataset by Fast-J cutoff pressure.
_, data = fns.split_pressure(data)

# Smaller subsample.
data = fns.sample(data)

# Pick out one time step.
#data = data[:, np.where(data[0] == 12.)].squeeze()

# Pick out one tropospheric level.
# About the top of Mt Everest.
# 9km / 85km ~= 10.6%
#trop = 0.10369537
#data = data[:, np.where(data[1] == trop)].squeeze()

# Pick out one column.
#data = data[:, np.where(data[2] == 51.875)].squeeze() # lat.
#data = data[:, np.where(data[3] == 359.0625)].squeeze() # lon.

# Input data.
inputs = data[phys_best]

inputs = np.swapaxes(inputs, 0, 1) 
print('\nInputs:', inputs.shape)

# Test all the targets.
for target_idx in [NO2]:  
  targets = data[target_idx]
  print('\nTarget:', target_idx, targets.shape)
  
  in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1, random_state=6, shuffle=False) 
  
  # Standardisation (optional).
  scaler = StandardScaler()
  in_train = scaler.fit_transform(in_train)
  in_test = scaler.fit_transform(in_test)

  # Find suitable size of Lasso regularisation const.
  avg = np.mean(out_train)
  e = np.floor(np.log10(avg)) - 1
  reg = 10**e
  
  # Train regression model.
  #model = linear_model.Lasso(reg)
  model = linear_model.LinearRegression()
  print(model)
  model.fit(in_train, out_train)
  
  # Test.
  # Should also implement k-fold cross validation.
  pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)
  
  # Make them the right shape.
  pred = pred.squeeze()
  out_test = out_test.squeeze()
  
  # Plot.
  fns.show(out_test, pred, maxe, mse, r2)
