'''
15/7/2025.
Check performance of random forest on stratospheric lookup table
portion of data, at pressures < 20 Pa.
'''

import time
import numpy as np
import functions as fns
import constants as con
from sklearn.metrics import r2_score


def portion_performance(portion_idxs, portion_info, rxn_idx, rxn_name):
   # Select the data above or below 20 Pa.
  inputs_portion = inputs[portion_idxs]
  targets_portion = targets[portion_idxs]
  preds_portion = preds[portion_idxs]
  # Loop through each column and get the average differences.
  print('\nCreating the grid for the column map.')
  grid = fns.make_cols_map(inputs_portion, targets_portion, preds_portion, rxn_idx) 
  if grid is not None:
    print('grid', grid.shape)
    # Get the average R2 score.
    targets_portion = targets_portion.ravel()
    preds_portion = preds_portion.ravel()
    r2 = round(r2_score(targets_portion, preds_portion), 3) 
    # Plot a map.
    print('\nPlotting the map.')
    fns.show_diff_map(rxn_name, grid, r2, portion_info)


# Load model data.
start = time.time()
inputs, targets, preds = fns.load_model_data('rf_full_range')
end = time.time()
elapsed = end - start
seconds = round(elapsed)
print(f'That took {seconds} seconds.')

# Select reactions that occur in the stratosphere.
# All, NO2, H2SO4, H2O, HOCl, O3
rxn_idxs = [None, 1, 13, 21, 24, 31]
rxn_names = ['all', f'NO{con.sub2}', 'sulfuric acid', 'water', 'HOCl', 'ozone']

# Get indices where pressures are above and below 20 Pa.
bottom_idxs = np.where(inputs[:, con.pressure] > 20)[0]
top_idxs = np.where(inputs[:, con.pressure] < 20)[0]
bottom_info = 'at pressures > 20 Pa.'
top_info = 'at pressures < 20 Pa.'

# For each of the reactions...
for rxn in range(2, len(rxn_idxs)):
  rxn_idx = rxn_idxs[rxn]
  rxn_name = rxn_names[rxn]
  print(f'\nLooking at {rxn_name}.')
  # Plot the performance at the portions.
  portion_performance(bottom_idxs, bottom_info, rxn_idx, rxn_name)
  portion_performance(top_idxs, top_info, rxn_idx, rxn_name)
  
# Print out the trop, strat and diff R2 scores for every J rate.  
for rxn in range(len(targets[0])):
  target_bottom = targets[bottom_idxs, rxn]
  pred_bottom = preds[bottom_idxs, rxn]
  target_top = targets[top_idxs, rxn]
  pred_top = preds[top_idxs, rxn]
  r2_bottom = round(r2_score(target_bottom, pred_bottom), 3)
  r2_top = round(r2_score(target_top, pred_top), 3) 
  r2_diff = round(abs(r2_bottom - r2_top), 3)
  print(f'\nReaction {rxn+1}')
  print(f'R2 at pressures > 20 Pa = {r2_bottom}')
  print(f'R2 at pressures < 20 Pa = {r2_top}')
  print(f'Difference between portions = {r2_diff}') 
