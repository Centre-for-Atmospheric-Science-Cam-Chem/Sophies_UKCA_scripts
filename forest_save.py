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
import file_paths as paths
import prediction_fns as fns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as scaler

# File paths.
data_path = f'{paths.npy}/low_res_yr_500k.npy'
out_name = 'rf_scaled'
out_path = f'{paths.mod}/{out_name}/{out_name}'
model_path = f'{out_path}.pkl' 
in_scale_path = f'{out_path}_in_scaler.pkl'
out_scale_path = f'{out_path}_out_scaler.pkl'
out_test_path = f'{out_path}_test_targets.npy'
pred_path = f'{out_path}_pred.npy'
meta_path = f'{out_path}_metadata.txt'

start = time.time()
print('Loading data')
data = np.load(data_path)
print(data.shape)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Input data.
inputs = data[con.phys_no_o3]
if inputs.ndim == 1:
  inputs = inputs.reshape(1, -1) 
inputs = np.swapaxes(inputs, 0, 1)
print('Inputs:', inputs.shape)

# Target data.
targets = data[con.J_core]
if targets.ndim > 1:
  targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape)

# Scale inputs.
in_scale = scaler()
inputs = in_scale.fit_transform(inputs)

# 90/10 train test split.
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.1, shuffle=False, random_state=con.seed)

# Scale training targets.
out_scale = scaler()
out_scale.fit(targets)
out_train = out_scale.transform(out_train)

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

# Reverse scaling on predictions.
out_pred = out_scale.inverse_transform(out_pred)

# Prepare directory.
os.mkdir(out_dir)

# Save the trained model and scalers.
start = time.time()
print('Saving random forest model and scalers.')
joblib.dump(model, model_path) 
joblib.dump(in_scale, in_scale_path)
joblib.dump(out_scale, out_scale_path)
end = time.time()
print(f'Saving the random forest model and scalers took {round(end-start)} seconds.')

# Save the test dataset.
start = time.time()
print('Saving test dataset.')
np.save(out_test_path, out_test)
np.save(pred_path, out_pred)
end = time.time()
print(f'Saving the test datasets took {round(end-start)} seconds.')

# Write metadata.
meta = f'Date: {datetime.date.today()}\n\
         {model_path}: random forest model, made using scikit-learn. Read into Python using joblib.\n\
         {in_scale_path}: standardising scaler to use on inputs from new datasets. Read into Python using joblib.\n\
	 {out_scale_path}: standardising scaler to reverse scaling of predictions from new datasets (not the predictions provided here). Read into Python using joblib.\n\
	 {out_test_path}: 2d numpy array of test targets dataset for the random forest, from a 90% train, 10% test split of the training data, of shape(samples, features).\n\
	 {pred_path}: 2d numpy array of predictions from the above test set, of shape(samples, features). Scaling has been reversed.\n\
	 Training data: {data_path}\n\
	 Data alterations: night times and upper stratosphere removed.\n\
	 Inputs: phys_no_o3. Hour of day, altitude km, latitude deg, longitude deg, days since 1/1/2015, humidity, cloud fraction, pressure Pa, cos solar zenith angle, upward shortwave flux, downward shortwave flux, upward longwave flux, downward longwave flux, temperature K.\n\
	 Targets: J_core. All strat-trop J rates which are not duplicate functional groups or all zero.\n\
	 Trees: {len(model.estimators_)}.\n\
	 Max leaves per tree: 100,000.\n\
	 Nodes per tree: {model.estimators_[0].tree_.node_count}.\n\
	 Tree depth: {model.estimators_[0].tree_.max_depth}.\n\
	 Max features per tree: 30% of data.\n\
	 Max samples per tree: 20% of data.\n\
	 Random seed for test split and forest creation: {con.seed}.'
meta = re.sub(' +', ' ', meta)
meta_file = open(meta_path, 'w')
meta_file.write(meta)
meta_file.close()
	 
