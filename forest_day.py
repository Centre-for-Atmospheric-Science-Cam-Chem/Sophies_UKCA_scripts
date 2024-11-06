'''
Name: Sophie Turner.
Date: 6/11/2024.
Contact: st838@cam.ac.uk.
Train a random forest on a low-resolution year and test it on a different dataset.
'''

import time
import numpy as np
import file_paths as paths
import constants as con
import prediction_fns_numpy as fns
from sklearn.ensemble import RandomForestRegressor

# File paths.
train_file = f'{paths.npy}/low_res_yr_5k.npy' 
test_file = f'{paths.npy}/20150715.npy'

print()
start = time.time()
print('Loading data')
train_data = np.load(train_file)
test_data = np.load(test_file)
print('Train data:', train_data.shape)
print('Test data:', test_data.shape)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Remove night times and stratosphere from test set (they should've already been removed from training set).
print('Removing upper stratosphere and night times.')
test_data, _ = fns.split_pressure(test_data)
test_data = test_data[:, np.where(test_data[10] > 0)].squeeze()

# Remove test set date from training set.
# Get integer portion of date-time for test set. That is the day of year.
print('\nRemoving test day from train data.')
test_day = round(test_data[4, 0])
# Find any samples in the training set which have that day of year.
i_day = np.where((train_data[4] >= test_day) & (train_data[4] < test_day + 1)) 
# Remove them from the training set.
train_data = np.delete(train_data, i_day, axis=1)

# Choose input and target data.
inputs = con.phys_all
targets = con.J_core
in_train = train_data[inputs]
out_train = train_data[targets]
in_test = test_data[inputs]
out_test = test_data[targets]

# Swap dimensions to make them compatible with sklearn.
in_train = np.swapaxes(in_train, 0, 1)
in_test = np.swapaxes(in_test, 0, 1)
out_train = np.swapaxes(out_train, 0, 1)
out_test = np.swapaxes(out_test, 0, 1)

# Make the regression model.
model = RandomForestRegressor(n_estimators=5, n_jobs=5, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)

# Train the model.
start = time.time()
print('Training model')
model.fit(in_train, out_train)
end = time.time()
print(f'Training the model took {round(end-start)} seconds.')

# Test.
start = time.time()
print('Testing model')
out_pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)
print('out_test:', out_test.shape)
print('out_pred:', out_pred.shape)
end = time.time()
print(f'Testing the model took {round(end-start)} seconds.')

# View performance.
fns.show(out_test, out_pred, maxe, mse, mape, smape, r2)
# Get co-ordinates of test set.
coords = test_data[0:5]
# View column.
fns.show_col(out_test, out_pred, coords, 19, 'O3', False)
