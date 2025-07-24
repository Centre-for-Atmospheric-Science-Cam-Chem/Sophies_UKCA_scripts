'''
Name: Sophie Turner.
Date: 13/1/2025.
Contact: st838@cam.ac.uk.
Plot avgs of J rate differences between target and prediction
for specific input variables and reactions.
'''

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import file_paths as paths
import constants as con
from sklearn.metrics import r2_score

# File paths.
mod_name = 'rf'
mod_path = f'{paths.mod}/{mod_name}/{mod_name}'
inputs_path = f'{mod_path}_test_inputs.npy' 
targets_path = f'{mod_path}_test_targets.npy'
preds_path = f'{mod_path}_pred.npy'

# Load data.
print('\nLoading data.')
inputs = np.load(inputs_path)
targets = np.load(targets_path)
preds = np.load(preds_path)
print('Inputs:', inputs.shape)
print('Targets:', targets.shape)
print('Preds:', preds.shape)
 
# Sub-sample data.
size = len(preds)
i = con.rng.integers(0, size, round(size/1000), dtype=np.int32)
inputs = inputs[i]
targets = targets[i]
preds = preds[i]
print('Inputs:', inputs.shape)
print('Targets:', targets.shape)
print('Preds:', preds.shape)  
 
size = len(preds)

# Indices of J rates in J_core which are prone to poorer prediction performance.
js = np.arange(0, 22, dtype=int)

# Convert altitude and SZA to more interpretable units.
alt = inputs[:, 1] * 85 # km.
inputs[:, 1] = alt
sza = np.arccos(inputs[:, 8]) * 180 / pi # Degrees.
inputs[:, 8] = sza

# For each input variable...
for i in range(len(con.input_names)): 
  if i == 11 or i == 12:
    continue 
  field = inputs[:, i]
  field_name = con.input_names[i]
  field_name_short = con.input_names_short[i]
  # Get input values.
  vals = np.unique(field) 
  print()
  
  # For each J rate...
  for j in range(len(js)):
    j_name = con.J_names[j]
    j_name_short = con.J_names_short[j]
    print(f'{j_name_short} by {field_name_short}')
    target = targets[:, js[j]]
    pred = preds[:, js[j]]
    
    # Get average J rate at each unique input value.
    target_avgs, pred_avgs = np.empty(len(vals)), np.empty(len(vals))
    for k in range(len(vals)):
      x = vals[k]
      target_inx = target[field == x]
      pred_inx = pred[field == x]
      target_avg = np.mean(target_inx)
      pred_avg = np.mean(pred_inx)  
      target_avgs[k] = target_avg
      pred_avgs[k] = pred_avg
      
    # Get R2 for this specific input var.  
    r2 = round(r2_score(target_avgs, pred_avgs), 3)
    # Don't imply that it is perfect.
    if r2 == 1:
      r2 = 0.999
     
    plt.figure(figsize=(10,5))
    # Best marker depends on amount of data.
    if i < 4: 
      plt.plot(vals, target_avgs, label='Mean target from Fast-J')
      plt.plot(vals, pred_avgs, label='Mean prediction from random forest')
    else:
      plt.scatter(vals, target_avgs, label='Mean target from Fast-J', alpha=0.2)
      plt.scatter(vals, pred_avgs, label='Mean prediction from random forest', alpha=0.2) 
         
    plt.title(f'Effect of {field_name} on {j_name} photolysis rate predictions by a random forest. R{con.sup2} = {r2}')
    plt.xlabel(field_name)
    plt.ylabel(f'J{j_name}')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig(f'{paths.analysis}/preds_per_input/{j_name_short}_mean_{field_name_short}.png')
    plt.close() 
    
