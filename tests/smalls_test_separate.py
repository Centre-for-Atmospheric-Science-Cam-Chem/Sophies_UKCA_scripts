'''
Name: Sophie Turner.
Date: 22/11/2024.
Contact: st838@cam.ac.uk.
Select single J rates from model trained on all and check their smallest J rates
to see where the random forest's performance drops with the smallest J rates.
'''

import joblib
import numpy as np
import matplotlib.pyplot as plt
import file_paths as paths
import constants as con
import prediction_fns_numpy as fns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score 

# File paths.
name = 'rf'
dir_path = f'{paths.mod}/{name}'
model_path = f'{dir_path}/{name}.pkl'
in_test_path = f'{dir_path}/{name}_test_inputs.npy'
out_test_path = f'{dir_path}/{name}_test_targets.npy'
pred_path = f'{dir_path}/{name}_pred.npy'
targets_path = f'{dir_path}/{name}_all_targets.npy'

print(f'\nLoading data from {name}.')

out_test = np.load(out_test_path) 
out_pred = np.load(pred_path) 

print('out test:', out_test.shape) # (18200000, 22)
print('out pred:', out_pred.shape) # (18200000, 22)

# Targets to look at.
js = [7,9,16,17,8,11,14,19,5,21,2,20] 

# Choose a target J rate.
for j in js:
  print(f'\nTarget index {con.J_core[j]}.')
  
  # Separate top and bottom portions.
  target = out_test[:, j]
  pred = out_pred[:, j]    
  thresh = 0.1
  top = max(target)
  split = top * thresh
  i_small = np.where(target <= split)
  small_target = target[i_small]
  small_pred = pred[i_small]
  large_target = np.delete(target, i_small)
  large_pred = np.delete(pred, i_small)
  
  # Get metrics. 
  r2 = round(r2_score(target, pred), 3)
  print(f'R2 = {r2}')
  r2 = round(r2_score(large_target, large_pred), 3)
  print(f'Largest {int((1-thresh)*100)}% of predictions: R2 = {r2}')
  r2 = round(r2_score(small_target, small_pred), 3)
  print(f'Smallest {int(thresh*100)}% of predictions: R2 = {r2}')
  
  # View performance.
  i = np.random.randint(0, len(large_target), 10000)
  plt.scatter(large_target[i], large_pred[i], label=f'Largest {int((1-thresh)*100)}% of range of prediction values.', alpha=0.05)
  i = np.random.randint(0, len(small_target), 10000)
  plt.scatter(small_target, small_pred, label=f'Smallest {int(thresh*100)}% of range of prediction values.', alpha=0.05)
  plt.title('ISON')
  plt.xlabel('J rate from UKCA')
  plt.ylabel('J rate from random forest')
  plt.legend()
  plt.show()
  plt.close()
  
print()
