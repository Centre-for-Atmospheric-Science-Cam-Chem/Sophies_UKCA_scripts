'''
Name: Sophie Turner.
Date: 11/6/2025.
Contact: st838@cam.ac.uk.
Compare random forest and Fast J in an area over time.
'''

import numpy as np
import constants as con
import functions as fns
import matplotlib.pyplot as plt

# Fetch the test datasets from my trained model.
inputs, targets, preds = fns.load_model_data('rf_trop')

# Define a column above Cambridge.
top = 1.1521739e-01 
cam_lat = 51.875
cam_lon = 0.9375

# Cut it down to ~6 months for easier viewing. 
days_end = 180

# Select the data in my Cambdrige area.
idx = np.where((inputs[:, con.alt] <= top) & \
               (inputs[:, con.lat] == cam_lat) & \
               (inputs[:, con.lon] == cam_lon) &
	       (inputs[:, con.days] <= days_end))[0]
inputs = inputs[idx].squeeze()
targets = targets[idx].squeeze()
preds = preds[idx].squeeze()

# J rates to look at. [idx, formatted name].
O3 = [0, f'O{con.sub3}']
NO2 = [1, f'NO{con.sub2}']
HCHOr = [3, 'HCHO (radical rxn)']
H2O2 = [25, f'H{con.sub2}O{con.sub2}']

# Plot the J rates from ML and FastJ.
x = inputs[:, con.days]
for J in [O3, NO2, HCHOr, H2O2]:
  idx = J[0]
  name = J[1]
  target = targets[:, idx].squeeze() 
  pred = preds[:, idx].squeeze()
  plt.plot(x, target, label=f'J{name} from Fast-J')
  plt.plot(x, pred, label=f'J{name} from random forest')
  plt.legend()
  plt.title(f'{name} photolysis rates above Cambridge')
  plt.xlabel('Days since 01/01/2015')
  plt.ylabel(f'J rate / {con.pers}')
  plt.show()
  plt.close()
