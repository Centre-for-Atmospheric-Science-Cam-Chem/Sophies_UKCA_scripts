'''
Name: Sophie Turner.
Date: 28/3/2025.
Contact: st838@cam.ac.uk.
Compare models trained with and without altitude.
'''
import numpy as np
import functions as fns
import constants as con
from sklearn.metrics import r2_score


def show_lvls(inputs, targets, preds):
  # Show some maps by pressure without training on altitude.
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
      name = f'J rates at model level {i+1}' 
      fns.show_diff_map(name, grid, r2)  


# Load model data.
inputs_alt, targets_alt, preds_alt = fns.load_model_data('rf_trop')
inputs_noalt, targets_noalt, preds_noalt = fns.load_model_data('rf_no_alt')
 
# Get R2 of all.
r2_alt = r2_score(targets_alt, preds_alt)
r2_noalt = r2_score(targets_noalt, preds_noalt)
print(f'R2 with altitude: {round(r2_alt, 3)}')
print(f'R2 without altitude: {round(r2_noalt, 3)}')

# Show some maps by altitude without training on altitude.
show_lvls(inputs_alt, targets_noalt, preds_noalt) 
