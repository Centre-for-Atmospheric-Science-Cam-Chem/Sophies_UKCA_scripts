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

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.

data = np.load(data_file)

# Input features.
#features = [1,7,8,9,10] # Best 5 from feature selection.
features = phys_all
# Input data.
inputs = data[features]

inputs = np.swapaxes(inputs, 0, 1) 
print('\nInputs:', inputs.shape)

# Test all the targets.
for target_idx in [HCHOm, NO2, O3, H2O2]:  
  target = data[target_idx]
  print('\nTarget:', target.shape)
  
  in_train, in_test, out_train, out_test = train_test_split(inputs, target, test_size=0.1, random_state=6, shuffle=False) 
 
  # Standardisation (optional).
  scaler = StandardScaler()
  in_train = scaler.fit_transform(in_train)
  in_test = scaler.fit_transform(in_test)

  # Find suitable size of Lasso regularisation const.
  avg = np.mean(out_train)
  e = np.floor(np.log10(avg)) - 1
  reg = 10**e
  
  # Train lasso regression.
  model = linear_model.Lasso(reg)
  #model = linear_model.LinearRegression()
  print(model)
  model.fit(in_train, out_train)
  
  # Test.
  pred, mse, mape, r2 = fns.test(model, in_test, out_test)
  
  # Make them the right shape.
  pred = pred.squeeze()
  out_test = out_test.squeeze()
  
  # Plotting this many datapoints is excessive and costly. Reduce it to 1%.
  length = len(pred)
  idxs = np.arange(0, length, 100)
  pred = pred[idxs]
  out_test = out_test[idxs]
  del(idxs)
  
  # Plot.
  fns.show(out_test, pred, mse, r2)
