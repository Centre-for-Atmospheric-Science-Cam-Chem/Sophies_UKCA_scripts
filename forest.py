f'''
Name: Sophie Turner.
Date: 9/9/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a random forest using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''

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
# Best physics inputs from feature selection.
phys_best = [1,7,8,9,10,14]
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
NO3 = 25
HOCl = 71
H2O2 = 74
O3 = 78 # O(1D) product.

data = np.load(data_file)

# Split the dataset by Fast-J cutoff pressure.
data, _ = fns.split_pressure(data)

# Input data.
inputs = data[phys_best]
inputs = np.swapaxes(inputs, 0, 1) 
print('\nInputs:', inputs.shape)

# Target data.
targets = data[HCHOr]
print('\nTargets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = train_test_split(inputs, target, test_size=0.05, random_state=6, shuffle=False) 
 
# Make the regression model.
# 5/15 features per tree.
# Remove or increase max_samples parameter if performance is poor.
model = RandomForestRegressor(n_estimators=500, max_features=0.3, n_jobs=4, random_state=6, max_samples=0.2)

# Cross-validate.
cv = KFold(random_state=6)
scores = cross_val_score(model, in_train, out_train, cv=cv)
print("Cross-validated scores (MSE):", -scores)

# Train the model.
model.fit(in_train, out_train)

# Get feature importance out.
ranks = model.feature_importances_
print('\nFeature importances:', ranks)

# Test.
out_pred, mse, mape, r2 = fns.test(model, in_test, out_test)

# Save the output data in case plt breaks on Conda again.
np.save(f'{paths.npy}/out_pred.npy', out_pred)
np.save(f'{paths.npy}/out_test.npy', out_test)

# View performance.
fns.show(out_test, out_pred, mse, r2)
