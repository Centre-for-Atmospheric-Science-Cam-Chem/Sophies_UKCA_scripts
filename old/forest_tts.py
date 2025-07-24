'''
Name: Sophie Turner.
Date: 9/9/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a random forest using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''

import time
import psutil
import joblib
import numpy as np
import file_paths as paths
import constants as con
import prediction_fns_numpy as fns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Memory usage at start of program.
GB = 1000000000
mem_start = psutil.virtual_memory().used / GB

# File paths.
data_file = f'{paths.npy}/low_res_yr_2015.npy'
#data_file = paths.four

print()
start = time.time()
print('Loading data')
data = np.load(data_file)
print(data.shape)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

'''
# Split the dataset by Fast-J cutoff pressure.
print('Removing upper stratosphere')
data, _ = fns.split_pressure(data)
# Remove zero flux.
print('Removing night times.')
data = data[:, np.where(data[10] > 0)].squeeze()
'''

# Input data.
inputs = data[con.phys_no_o3]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[[con.NO2, con.HCHOr]]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# TTS.
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1)

# Make the regression model.
#model = RandomForestRegressor(n_estimators=5, n_jobs=5, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)
model = RandomForestRegressor(n_estimators=20, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)

'''
# Cross-validate. Use on smaller data or a smaller forest.
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
'''
# Save the output data in case plt breaks on Conda again.
start = time.time()
print('Saving output')
np.save(f'{paths.npy}/out_pred.npy', out_pred)
np.save(f'{paths.npy}/out_test.npy', out_test)
end = time.time()
print(f'Saving the output took {round(end-start)} seconds.')
'''
'''
# Save the trained model.
start = time.time()
print('Saving random forest model')
joblib.dump(model, f'{paths.npy}/RF5.pkl') 
end = time.time()
print(f'Saving the random forest model took {round(end-start)} seconds.')
'''
# View performance.
fns.show(out_test, out_pred, maxe, mse, mape, smape, r2)

# Memory usage at end.
mem_end = psutil.virtual_memory().used / GB
mem = round(mem_end - mem_start, 3)
print(f'Memory usage at the end of the program: {mem} GB.')
