'''
Name: Sophie Turner.
Date: 10/5/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with linear regression using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import glob
import numpy as np
import matplotlib.pyplot as plt
import prediction_fns_numpy as fns
from sklearn.preprocessing import StandardScaler
  
    
def test(in_train, in_test, out_train, out_test):
  # Use 2 different datasets for train and test.
  # Reshape the targets so they enter ML functions in the right order.
  out_train = np.rot90(out_train, 3)
  out_test = np.rot90(out_test, 3)
  # Linear regression.
  model = fns.train(in_train, out_train)
  pred, mse, mape, r2 = fns.test(model, in_test, out_test)
  print('R2:', r2)
  print('Mean squared err:', mse)
  print('Mean absolute percentage er:', mape)
  return(pred)
  
  
# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
train_files = glob.glob(f'{dir_path}/2015*15.npy')
train_file = f'{dir_path}/4days.npy'
test_file = f'{dir_path}/20170415.npy'
name_file = f'{dir_path}/idx_names'

print(test_file)

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.linspace(0,13,14, dtype=int)
J_all = np.linspace(14,83,70, dtype=int)
NO2 = 15
HCHOr = 17 # Radical product.
HCHOm = 18 # Molecular product.
H2O2 = 73
O3 = 77 # O(1D) product.

# Names of the fields. See metadata.txt.
idx_names = fns.get_idx_names(name_file)
  
# Have a look at the fields to choose which ones to use.  
#for name in idx_names:
#  print(name) 

# Get the training data.
train_days = fns.collate(train_file, train_files) 

# Testing data.
test_day = np.load(test_file)

# Input features.
features = [1,7,8,9,10]
# Input data.
in_train = train_days[features]
in_test = test_day[features]

# Reshape the arrays so they enter ML functions in the right order however many fields are selected.
in_train, in_test = fns.shape(in_train), fns.shape(in_test)
in_train, in_test = np.rot90(in_train, 3), np.rot90(in_test, 3)

# Standardisation (optional).
scaler = StandardScaler()
in_train = scaler.fit_transform(in_train)
in_test = scaler.fit_transform(in_test)

# Test all the targets.
for target_idx in [HCHOm, NO2, O3, H2O2]:  
  out_train = train_days[target_idx]
  out_test = test_day[target_idx]
  # Reshape the arrays so they enter ML functions in the right order.
  out_train, out_test = fns.shape(out_train), fns.shape(out_test)
  # Do linear regression and get results.
  pred = test(in_train, in_test, out_train, out_test)
  
  # Make them the right shape.
  pred = pred.squeeze()
  out_test = out_test.squeeze()
  
  # Plotting this many datapoints is excessive and costly. Reduce it to 1%.
  length = len(pred)
  idxs = np.arange(0, length, 100)
  pred = pred[idxs]
  out_test = out_test[idxs]
  del(idxs)
  
  # Show a plot of results.
  plt.scatter(out_test, pred, alpha=0.1)
  plt.title('J rates on 15/4/17')
  plt.xlabel('targets from UKCA')
  plt.ylabel('predictions by linear regression')
  plt.show()
