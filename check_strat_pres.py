'''
15/7/2025.
Check performance of random forest on stratospheric lookup table
portion of data, at pressures < 20 Pa.
'''

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
  grid = fns.make_cols_map(inputs_portion, targets_portion, preds_portion, rxn_idx) 
  # Get the average R2 score.
  targets_portion = targets_portion.ravel()
  preds_portion = preds_portion.ravel()
  r2 = round(r2_score(targets_portion, preds_portion), 3) 
  # Plot a map.
  fns.show_diff_map(rxn_name, grid, r2, portion_info)


# Load model data.
inputs, targets, preds = fns.load_model_data('rf')

# Select reactions that occur in the stratosphere.
# NO2, HOCl, O3
rxn_idxs = [16, 71, 78]
rxn_names = [f'NO{con.sub2}', 'HOCl', 'O{con.sub3}']

# Get indices where pressures are above and below 20 Pa.
bottom_idxs = np.where(inputs[:, con.pressure] > 20)[0]
top_idxs = np.where(inputs[:, con.pressure] < 20)[0]
bottom_info = 'at pressures > 20 Pa.'
top_info = 'at pressures < 20 Pa.'

# For each of the reactions...
for rxn in range(len(rxn_idxs)):
  rxn_idx = rxn_idxs[rxn]
  rxn_name = rxn_names[rxn]
  # Plot the performance at the portions.
  portion_performance(bottom_idxs, bottom_info, rxn_idx, rxn_name)
  portion_performance(top_idxs, top_info, rxn_idx, rxn_name)
