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
import psutil
import numpy as np
import file_paths as paths
import prediction_fns_numpy as fns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Memory usage at start of program.
GB = 1000000000
mem_start = psutil.virtual_memory().used / GB

# File paths.
#data_file = f'{paths.npy}/20150115.npy'
data_file = f'{paths.npy}/4days.npy'
name_file = f'{paths.npy}/idx_names'

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=np.int16)
J_all = np.arange(15, 85, dtype=np.int16)
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
NO3 = 66
HOCl = 71
H2O2 = 74
O3 = 78 # O(1D) product.
# Physics inputs chosen by feature selection.
phys_best = [1, 7, 8, 9, 10, 14]
# J rates which are not summed or duplicate fg. with usually zero rates removed.
J_core = [16,18,19,20,24,28,30,31,32,33,51,52,66,70,71,72,73,74,75,78,79,80,81,82]
# For consistent random states.
seed = 6

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
inputs = data[phys_best]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[J_core]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.05, random_state=seed, shuffle=False) 
 
# Make the regression model.
# 5/15 features per tree.
#model = RandomForestRegressor(n_estimators=1, random_state=seed, max_leaf_nodes=200000)
model = RandomForestRegressor(n_estimators=2000, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100, random_state=seed)

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

# Memory usage after training model.
mem_mid = psutil.virtual_memory().used / GB
mem = round(mem_mid - mem_start, 3)
print(f'Memory usage after training model: {mem} GB.')

'''
# Get feature importance out.
ranks = model.feature_importances_
print('Feature importances:')
for rank in ranks:
  print(rank)
'''

# Get info about the trees in our forest for explainability.
print(model)
for i in range(len(model.estimators_)):
  tree = model.estimators_[i]
  nodes = tree.tree_.node_count
  depth = tree.tree_.max_depth
  feature = tree.tree_.feature[0]
  thresh = tree.tree_.threshold[0]
  print(f'Tree {i+1} has {nodes} nodes and is {depth} branches tall.')
  print(f'Tree {i+1} begins splitting on feature {feature} at a threshold of {thresh}.')

# Test.
start = time.time()
print('Testing model')
out_pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)
print('out_test:', out_test.shape)
print('out_pred:', out_pred.shape)
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
fns.show(out_test, out_pred, maxe, mse, r2)

# Memory usage at end.
mem_end = psutil.virtual_memory().used / GB
mem = round(mem_end - mem_start, 3)
print(f'Memory usage at the end of the program: {mem} GB.')
