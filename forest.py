f'''
Name: Sophie Turner.
Date: 9/9/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a random forest using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''

import time
import joblib
import numpy as np
import file_paths as paths
import prediction_fns_numpy as fns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# File paths.
data_file = f'{paths.npy}/4days.npy'
name_file = f'{paths.npy}/idx_names'

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
NO3 = 25
HOCl = 71
H2O2 = 74
O3 = 78 # O(1D) product.
# J rates which are not summed or duplicate fg. with usually zero rates removed.
J_core = [16,18,19,20,24,25,28,30,31,32,33,51,52,66,68,70,71,72,73,74,75,76,78,79,80,81,82]

print()
start = time.time()
print('Loading data')
data = np.load(data_file)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Split the dataset by Fast-J cutoff pressure.
print('Removing upper stratosphere')
data, _ = fns.split_pressure(data)

# Input data.
inputs = data[8] # SZA
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[H2O2]
#targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.05, random_state=6, shuffle=False) 
 
# Make the regression model.
# 5/15 features per tree.
# Remove or increase max_samples parameter if performance is poor.
model = RandomForestRegressor(n_estimators=50, max_features=0.3, n_jobs=4, random_state=6, max_samples=0.2)

'''
# Cross-validate. Use on smaller data or with fewer trees.
start = time.time()
print('Cross-validating.')
scores = cross_val_score(model, in_train, out_train, cv=KFold())
print("Cross-validated scores (MSE):", scores)
end = time.time()
print(f'Cross-validation took {round(end-start)} seconds.')
'''

# Train the model.
start = time.time()
print('Training model')
model.fit(in_train, out_train)
end = time.time()
print(f'Training the model took {round(end-start)} seconds.')

# Get feature importance out.
ranks = model.feature_importances_
print('Feature importances:')
for rank in ranks:
  print(rank)

# Test.
start = time.time()
print('Testing model')
out_pred, mse, mape, r2 = fns.test(model, in_test, out_test)
end = time.time()
print(f'Testing the model took {round(end-start)} seconds.')

# Save the output data in case plt breaks on Conda again.
start = time.time()
print('Saving output')
np.save(f'{paths.npy}/out_pred.npy', out_pred)
np.save(f'{paths.npy}/out_test.npy', out_test)
end = time.time()
print(f'Saving the output took {round(end-start)} seconds.')
'''
# Save the trained model.
start = time.time()
print('Saving random forest model')
joblib.dump(model, f'{paths.npy}/RF5.pkl') 
end = time.time()
print(f'Saving the random forest model took {round(end-start)} seconds.')
'''
# View performance.
fns.show(out_test, out_pred, mse, r2)
