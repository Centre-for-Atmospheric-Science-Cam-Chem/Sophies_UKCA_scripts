'''
Name: Sophie Turner.
Date: 14/11/2024.
Contact: st838@cam.ac.uk.
See where the random forest's performance drops with the smallest J rates.
'''

import time
import joblib
import numpy as np
import file_paths as paths
import constants as con
import prediction_fns_numpy as fns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score 

# File paths.
model_name = 'rf'
model_file = f'{paths.npy}/{model_name}.pkl'
in_test_file = f'{paths.npy}/{model_name}_test_inputs.npy' 
out_test_file = f'{paths.npy}/{model_name}_test_targets.npy'
targets_file = f'{paths.npy}/{model_name}_all_targets.npy'
data_file = f'{paths.npy}/low_res_yr_500k.npy'

print('Loading model and test data.')
start = time.time()
model = joblib.load(model_file)
in_test = np.load(in_test_file)
out_test = np.load(out_test_file)
targets = np.load(targets_file)
end = time.time()
print(f'Loading the model and test data took {round(end-start)} seconds.')
print('in_test:', in_test_p.shape)
print('out_test:', out_test_p.shape)

tops = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for top in tops:
  bottom = top - 0.1
  print(f'\nTrained on all but test on all except the {int(bottom * 100)}% to {int(top * 100)}% of J rates\n')
  
  # Test on the smallest J rates.
  in_test_p, out_test_p, deletes = fns.only_range(in_test, out_test, targets, bottom, top, False)
  print('in_test:', in_test_p.shape)
  print('out_test:', out_test_p.shape)
  
  exit()
  
  # Test.
  start = time.time()
  print('Testing model')
  out_pred = model.predict(in_test_p)
  # Remove predictions which can't be compared.
  #out_pred = np.delete(out_pred, deletes, axis=1)
  mape = mean_absolute_percentage_error(out_test_p, out_pred)
  r2 = round(r2_score(out_test_p, out_pred), 3)

  # View performance.
  fns.show(out_test_p, out_pred, 0, 0, mape, 0, r2)
