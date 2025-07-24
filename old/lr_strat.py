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
lower, upper = fns.split_pressure(data)

# Training input data.
in_train = lower[phys_best]
in_train = np.swapaxes(in_train, 0, 1) 
print('\nTraining inputs:', in_train.shape)

# The inputs to use the trained model on.
in_use = upper[phys_best]
in_use = np.swapaxes(in_use, 0, 1) 
print('\nInputs for prediction:', in_train.shape)

# Standardisation (optional).
scaler = StandardScaler()
in_train = scaler.fit_transform(in_train)
in_use = scaler.fit_transform(in_use)

# Test all the targets.
for target_idx in [HCHOm]:  
  out_train = lower[target_idx]
  print('\nTraining target:', target_idx, out_train.shape)
  
  # Train regression model.
  #model = linear_model.Lasso(reg)
  model = linear_model.LinearRegression()
  print(model)
  model.fit(in_train, out_train)
  
  # Use the model on lookup-table portion of atmosphere.
  pred = model.predict(in_use)
  
  # Lookup table J rate.
  lookup = upper[target_idx]
 
  # Dimension for plot.
  dim = upper[8] # sza.
 
  # Plotting this many datapoints is excessive and costly. Reduce it to 1%.
  length = len(pred)
  idxs = np.arange(0, length, 100)
  pred = pred[idxs]
  lookup = lookup[idxs]
  dim = dim[idxs]
  del(idxs)
  
  plt.scatter(dim, pred,  label='Predictions from linear model trained on Fast-JX below cut-off')
  plt.scatter(dim, lookup,  label='Lookup table')  
  plt.legend()
  plt.title(f'HCHO -> CO + H\u2082 above Fast-JX cut-off')
  #plt.xlabel('Downward shortwave flux / Wm\u207b\u00b2')
  plt.xlabel('J rate / \u207b\u00b9')
  plt.ylabel('Solar zenith angle / cos(radians)')
  plt.show() 
  plt.close()
