'''
Name: Sophie Turner.
Date: 11/6/2025.
Contact: st838@cam.ac.uk.
Compare random forest and Fast J in an area over time.
'''

import joblib
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt

# Fetch my trained model.
rf_path = f'{paths.mod}/rf_trop/rf_trop.pkl'
rf = joblib.load(rf_path)

# Fetch a test day of full-res data.
data_path = f'{paths.npy}/20161001.npy'
data = np.load(data_path)

# Get the data in the right shape for ML.
data = np.swapaxes(data, 0, 1)

# Define a gridbox above Cambridge.
cam_alt = 1.0839497e-02
cam_lat = 51.875
cam_lon = 0.9375

# Select the test data in my Cambdrige box.
idx = np.where((data[:, con.alt] == cam_alt) & \
               (data[:, con.lat] == cam_lat) & \
               (data[:, con.lon] == cam_lon))[0]
data = data[idx].squeeze()

# Get inputs and targets.
inputs = data[:, [0,1,2,3,4,5,6,7,8,9,10,13]]

# Use my model on the inputs.
preds = rf.predict(inputs)

# J rates to look at. [pred idx, target idx, formatted name].
O3 = [0, 15, f'O{con.sub3}'] 
NO2 = [1, 16, f'NO{con.sub2}'] 
HCHOr = [3, 18, 'HCHO (radical rxn)'] 
H2O2 = [25, 74, f'H{con.sub2}O{con.sub2}']

# Plot the J rates from ML and FastJ.
x = inputs[:, con.hour]
x[-1] = 24
for J in [O3, NO2, HCHOr, H2O2]:
  pred_idx = J[0]
  target_idx = J[1]
  name = J[2]
  target = data[:, target_idx].squeeze() 
  pred = preds[:, pred_idx].squeeze()
  plt.plot(x, target, label=f'J{name} from Fast-J')
  plt.plot(x, pred, label=f'J{name} from random forest')
  plt.xlim((0, 24))
  plt.legend()
  plt.title(f'{name} photolysis rates 1 km above Cambridge on 1st Oct 2016')
  plt.xlabel('Hour of day, UTC')
  plt.ylabel(f'J rate / {con.pers}')
  plt.show()
  plt.close()
