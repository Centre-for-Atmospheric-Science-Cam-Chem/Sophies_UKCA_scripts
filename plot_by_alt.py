'''
Name: Sophie Turner.
Date: 21/3/2025.
Contact: st838@cam.ac.uk.
Test effect of SZA and altitude on accuracy.
'''
import numpy as np
import functions as fns
import constants as con
from sklearn.metrics import r2_score

# Load model data.
inputs, targets, preds = fns.load_model_data('rf_trop')

# Get average of all.
r2 = round(r2_score(targets, preds), 3) 
# Loop through each column and get the average differences.
grid = fns.make_cols_map(inputs, targets, preds)
# Plot a map.
name = 'all' 
print(r2)
fns.show_diff_map(name, grid, r2)

# Pick out the altitudes/pressures.
alts = np.unique(inputs[:, con.alt])
# For each altitude/pressure...
for i in range(len(alts)):
  alt = alts[i]
  # Select the data at this level.
  lvl = np.where(inputs[:, con.alt] == alt)[0]
  input, target, pred = inputs[lvl], targets[lvl], preds[lvl]
  # Loop through each column and get the average differences.
  grid = fns.make_cols_map(input, target, pred) 
  # Get the average R2 score.
  target = target.ravel()
  pred = pred.ravel()
  r2 = round(r2_score(target, pred), 3) 
  print(r2)  
  # Plot a map once every 10 levels.
  if (i+10)%10 == 0:
    name = f'model level {i+1}' 
    fns.show_diff_map(name, grid, r2)
