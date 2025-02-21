'''
Name: Sophie Turner.
Date: 20/11/2024.
Contact: st838@cam.ac.uk.
Train a standard random forest and save it and its test data.
'''

import os
import re
import time
import joblib
import datetime
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler as scaler

# Whether or not to standardise data.
scale = True

# File paths.
data_path = f'{paths.npy}/low_res_yr_500k.npy'
if scale:
  out_name = 'rf_scaled'
else:
  out_name = 'rf'
out_dir = f'{paths.mod}/{out_name}'
out_path = f'{out_dir}/{out_name}'
model_path = f'{out_path}.pkl' 
in_scale_path = f'{out_path}_in_scaler.pkl'
out_scale_path = f'{out_path}_out_scaler.pkl'
out_test_path = f'{out_path}_targets.npy'
pred_path = f'{out_path}_pred.npy'
in_test_path = f'{out_path}_inputs.npy'
meta_path = f'{out_path}_metadata.txt'

start = time.time()
print('\nLoading data')
data = np.load(data_path)
print(data.shape)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, con.phys_main, con.J_all)

# Scale inputs.
if scale:
  in_scale = scaler()
  inputs = in_scale.fit_transform(inputs)
  
# 90/10 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets)

# Scale training targets except for NO3.
if scale:
  out_scale = scaler()
  out_scale.fit(targets)
  #NO3 = out_train[:, 11].copy()
  out_train = out_scale.transform(out_train)
  #out_train[:, 11] = NO3

# Make the regression model.
model = RandomForestRegressor(n_estimators=20, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)

# Train the model.
start = time.time()
print('Training model')
model.fit(in_train, out_train)
end = time.time()
print(f'Training the model took {round(end-start)} seconds.')

# Test the model.
print('Testing the model.')
out_pred = model.predict(in_test) 

# Reverse scaling on test inputs and predictions, except for NO3, which was not scaled.
if scale:
  in_test = in_scale.inverse_transform(in_test)
  #NO3 = out_pred[:, 11].copy()
  out_pred = out_scale.inverse_transform(out_pred)
  #out_pred[:, 11] = NO3

# Prepare directory.
if not os.path.exists(out_dir):
  os.mkdir(out_dir)

# Save the trained model, data and scalers.
start = time.time()
print('Saving random forest model, and scalers if chosen.')
joblib.dump(model, model_path) 
if scale:
  joblib.dump(in_scale, in_scale_path)
  joblib.dump(out_scale, out_scale_path)
end = time.time()
print(f'Saving the random forest model, and scalers if chosen, took {round(end-start)} seconds.')

# Save the test dataset.
start = time.time()
print('Saving test dataset.')
np.save(out_test_path, out_test)
np.save(pred_path, out_pred)
np.save(in_test_path, in_test)
end = time.time()
print(f'Saving the test datasets took {round(end-start)} seconds.')

# Metadata text.
if scale:
  text = f'{in_scale_path}: standardising scaler to use on inputs from new datasets. Scaling of provided inputs has been reversed already. Read into Python using joblib.\n\
{out_scale_path}: standardising scaler to reverse scaling of predictions from new datasets (not the predictions provided here). Scaling of priovided predictions has been reversed already. Read into Python using joblib.\n'
else:
  text = ''

# Write metadata.
meta = f'Date: {datetime.date.today()}\n\
{model_path}: random forest model, made using scikit-learn. Read into Python using joblib.\n\
{text}\
{out_test_path}: 2d numpy array of test targets dataset for the random forest, from a 90% train, 10% test split of the training data, of shape(samples, features).\n\
{pred_path}: 2d numpy array of predictions from the above test set, of shape(samples, features).\n\
{in_test_path}: 2d numpy array of inputs used to make the above datasets, of shape(samples, features).\n\
Training data: {data_path}\n\
Data alterations: night times and upper stratosphere removed.\n\
Inputs: phys_main. Hour of day, altitude km, latitude deg, longitude deg, days since 1/1/2015, humidity, cloud fraction, pressure Pa, cos solar zenith angle, upward shortwave flux, downward shortwave flux, temperature K.\n\
Targets: J_core. All strat-trop J rates which are not duplicate functional groups or all zero.\n\
Trees: {len(model.estimators_)}.\n\
Max leaves per tree: 100,000.\n\
Nodes per tree: {model.estimators_[0].tree_.node_count}.\n\
Tree depth: {model.estimators_[0].tree_.max_depth}.\n\
Max features per tree: 30% of data.\n\
Max samples per tree: 20% of data.\n\
Random seed for test split and forest creation: {con.seed}.'
meta_file = open(meta_path, 'w')
meta_file.write(meta)
meta_file.close() 
