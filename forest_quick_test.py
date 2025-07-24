'''
21/7/2025
A quick, basic overview of a saved random forest's performance.
'''

import os
import idx_names
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.metrics import r2_score

model_name = 'rf_trop'
  
# Load the model data.
_, targets, preds = fns.load_model_data(model_name)

# Overall average R2 score for all J rates.
r2 = round(r2_score(targets, preds), 3)
print(f'\nOverall average {con.r2} score = {r2}\n')

# Print out the trop, strat and diff R2 scores for every J rate.  
lows = []
for rxn in range(len(targets[0])):
  name = idx_names.idx_names_trop[rxn + 15][2]
  target = targets[:, rxn]
  pred = preds[:, rxn]
  r2 = round(r2_score(target, pred), 3)
  print(f'{name} {con.r2} = {r2}')
  if r2 < 0.95:
    lows.append(name)
    
# See where the problems are.    
print(f'\nLow-scoring J rates: {lows}')
