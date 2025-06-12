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


def hour_medians(inputs, outputs):
  '''Build up an array of medians and IQR for each hour of the day
     in a dataset.
     parameters:
     inputs (2D np array): input dataset. 
     outputs (1D np array): targets or predictions.
     returns:
     meds (1D array): medians for each hour.
     q25 (1D array): 25th precentile for each hour.
     q75 (1D array): 75th percentil for each hour.
  '''
  meds = []
  q25s = []
  q75s = []
  hours = np.arange(24)
  for hour in hours:
    idx = np.where(inputs[:, con.hour] == hour)[0]
    if len(idx) != 0:
      output = outputs[idx].squeeze()
      med = np.median(output)
      q25 = np.percentile(output, 25)
      q75 = np.percentile(output, 75)
      meds.append(med)
      q25s.append(q25)
      q75s.append(q75)
    else: 
      meds.append(0)
      q25s.append(0)
      q75s.append(0)
  return(meds, q25s, q75s)

# Fetch the test datasets from my trained model.
inputs, targets, preds = fns.load_model_data('rf_trop')

# Define a grid around Cambridge.
lat_min = 50.625 
lat_max = 53.125
lon_min = 359.0625 
lon_max = 2.8125
top = 1.1521739e-01

# Choose time of year.
jan = 0
feb = 30
mar = 60
apr = 90
may = 120
jun = 150
jul = 180
aug = 210
sep = 240
oct = 270
nov = 300
dec = 330
days_start = jun 
days_end = sep

# Select the data in my Cambdrige area.
idx = np.where((inputs[:, con.lat] >= lat_min) & \
               (inputs[:, con.lat] <= lat_max) & \
               (inputs[:, con.lon] <= lon_max) & \
	       (inputs[:, con.days] >= days_start) & \
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
x = np.arange(24)
for J in [O3, NO2, HCHOr, H2O2]:
  idx = J[0]
  name = J[1]
  target = targets[:, idx].squeeze() 
  pred = preds[:, idx].squeeze()

  # Get medians and IQRs for each hour of day in the year.
  med_target, q25_target, q75_target = hour_medians(inputs, target)
  med_pred, q25_pred, q75_pred = hour_medians(inputs, pred)
  
  # Draw the plot.
  plt.fill_between(x, q25_target, q75_target, color='palegreen', alpha=0.3, label=f'IQR of J{name} from Fast-J')
  plt.fill_between(x, q25_pred, q75_pred, color='lightsalmon', alpha=0.3, label=f'IQR of J{name} from random forest')
  plt.plot(x, med_target, color='green', label=f'Median of J{name} from Fast-J')
  plt.plot(x, med_pred, color='orangered', label=f'Median of J{name} from random forest')
  plt.legend()
  plt.title(f'Median {name} photolysis rates above Cambridge at each hour of the day in the Summer of 2015')
  plt.xlabel('Hour of day, UTC')
  plt.ylabel(f'J rate / {con.pers}')
  plt.show()
  plt.close()
