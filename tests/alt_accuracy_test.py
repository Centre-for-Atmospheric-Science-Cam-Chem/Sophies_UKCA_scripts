'''
Name: Sophie Turner.
Date: 24/3/2025.
Contact: st838@cam.ac.uk.
Compare random forest with and without altitude as an input.
'''
import numpy as np
import functions as fns
import constants as con
from idx_names import idx_names
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load original random forest test data.
inputs_orig, targets_orig, preds_orig = fns.load_model_data('rf_trop')

# Load test data for random forest without altitude.
inputs_noalt, targets_noalt, preds_noalt = fns.load_model_data('rf_no_alt')

# Shrink dataset.
targets_orig, preds_orig, alpha = fns.shrink(targets_orig, preds_orig)
targets_noalt, preds_noalt, alpha = fns.shrink(targets_noalt, preds_noalt)

# Compare R2.
r2_orig = round(r2_score(targets_orig, preds_orig), 3)
r2_noalt = round(r2_score(targets_noalt, preds_noalt), 3)
print('R2 with altitude:', r2_orig)
print('R2 without altitude:', r2_noalt)

# Show a pred vs target plot for each J rate.
for j in range(0, len(targets_orig[0]), 10):
  name = idx_names[con.J_trop[j]][2]
  target_orig = targets_orig[:, j]
  pred_orig = preds_orig[:, j]
  target_noalt = targets_noalt[:, j]
  pred_noalt = preds_noalt[:, j]
  r2_orig = round(r2_score(target_orig, pred_orig), 3)
  r2_noalt = round(r2_score(target_noalt, pred_noalt), 3)

  plt.scatter(target_orig, pred_orig, alpha=alpha, label=f'With altitude, R2 = {r2_orig}')
  plt.scatter(target_noalt, pred_noalt, alpha=alpha, label=f'Without altitude, R2 = {r2_noalt}')
  plt.title(name)
  plt.xlabel('Target J rate')
  plt.ylabel('Predicted J rate')
  plt.legend() 
  plt.show()
  plt.close()
  
