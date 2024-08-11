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
  
  # Train lasso regression.
  model = linear_model.Lasso(0.01)
  #model = linear_model.LinearRegression()
  print(model)
  model.fit(in_train, out_train)
  
  # Test.
  pred, mse, mape, r2 = fns.test(model, in_test, out_test)
  print('R2:', r2)
  print('MSE:', mse)
  
