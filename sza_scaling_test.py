'''
Name: Sophie Turner.
Date: 31/3/2025.
Contact: st838@cam.ac.uk.
Compare random forests with and without scaling solar zenith angle with vertical height.
Found no significant change from scaling SZA.
'''

import numpy as np
import functions as fns
import constants as con
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def compare_sza_scaling(model_name, title):
  '''Look at performance of random forest trained of surface level, 
  vertically scaled or pressure scaled solar zenith angle (SZA).
  model_name (string): the name of the random forest to use as it is named in its directory.
  title (string): description of type of SZA scaling used.
  '''
  # Load model data.
  inputs, targets, preds = fns.load_model_data(model_name)
  # Get R2 of all.
  r2 = r2_score(targets, preds)
  print(f'R2 with {title} SZA: {round(r2, 3)}') 
  
  # See a column map at middays.
  midday = inputs[:, con.hour] == 12
  inputs = inputs[midday]
  targets = targets[midday]
  preds = preds[midday]
 
  # See a column map of average performace on all J rates. 
  r2 = round(r2_score(targets, preds), 3)
  grid = fns.make_cols_map(inputs, targets, preds)
  fns.show_diff_map('all', grid, r2)
  
  # See a column map of all H2O data.
  targets = targets[:, 20]
  preds = preds[:, 20]
  r2 = round(r2_score(targets, preds), 3)
  grid = fns.make_cols_map(inputs, targets, preds)
  fns.show_diff_map('water', grid, r2)


# Look at performance of random forest trained of surface level SZA.
compare_sza_scaling('rf_trop', 'surface-level')
# Look at performance of random forest trained on scaled SZA.
compare_sza_scaling('rf_scaled_sza', 'vertically scaled')
