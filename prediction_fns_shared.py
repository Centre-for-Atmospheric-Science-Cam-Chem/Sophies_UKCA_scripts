'''
Name: Sophie Turner.
Date: 19/1/2023.
Contact: st838@cam.ac.uk
Functions which are used by prediction scripts.
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import psutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_percentage_error, r2_score 
  
# ML functions.  
  
def train(in_train, out_train):
  # Set up simple model.
  model = linear_model.LinearRegression()
  model.fit(in_train, out_train)
  return(model)


def sMAPE(out_test, out_pred):
  # Compute the symmetric mean absolute percentage error.
  # Like MAPE but handles values close to zero better.
  # test: array of test targets. pred: array of prediction outputs. 
  warnings.filterwarnings('ignore') 
  n = len(out_test)
  diffs = 0
  for i in range(n):
    diffs += (abs(out_test[i] - out_pred[i])) / (out_test[i] + out_pred[i])
  smape = (100 / n) * diffs 
  return(smape)
  
  
def metrics_2d(out_test, out_pred):
  # Get metrics over 2 dimensions for metrics which can only be used on 1 dim.
  # out_test, out_pred: 2D arrays.
  n = len(out_test[0])
  maxes = [] # Array of max errors for each target.
  smape = 0 # Average percentage error.
  for target in range(n):
    target_test = out_test[:,target].squeeze()
    target_pred = out_pred[:,target].squeeze()
    maxe = max_error(target_test, target_pred)
    maxes.append(maxe)
    smape += sMAPE(target_test, target_pred)
  smape = smape / n
  return(maxes, smape)
  

def test(model, in_test, out_test):
  # Try it out.
  out_pred = model.predict(in_test)
  # See how well it did.
  if out_test.ndim == 1:
    maxe = max_error(out_test, out_pred)
    smape = sMAPE(out_test, out_pred)
  else:
    maxe, smape = metrics_2d(out_test, out_pred)
  mse = mean_squared_error(out_test, out_pred)
  mape = mean_absolute_percentage_error(out_test, out_pred)
  r2 = round(r2_score(out_test, out_pred), 3)
  return(out_pred, maxe, mse, mape, smape, r2)  
  
  
# Results functions.  

def shrink(out_test, out_pred):
  # Don't plot an unnecessary number of data points i.e. >10000.  
  # Make them the right shape.
  out_test = out_test.squeeze()
  out_pred = out_pred.squeeze()
  length = len(out_pred)
  if length > 10000:
    # Plotting this many datapoints is excessive and costly. Reduce it to 10000.
    idxs = np.int16(np.linspace(0, length-1, 10000))
    out_test = out_test[idxs]
    out_pred = out_pred[idxs]
    del(idxs)
    # Choose opacity of points.
    alpha = 0.05
  elif length > 100:
    alpha = 0.1
  else:
    alpha = 1
  return(out_test, out_pred, alpha)


def force_axes():
  # Make plot axes exactly the same.
  plt.axis('square')
  xticks, xlabels = plt.xticks()
  yticks, ylabels = plt.yticks()
  plt.axis('auto')
  if len(yticks) > len(xticks):
    tix = yticks
    lab = ylabels
  else:
    tix = xticks
    lab = xlabels
  plt.xticks(ticks=tix, labels=lab)
  plt.yticks(ticks=tix, labels=lab) 
  
  
def show(out_test, out_pred, maxe, mse, mape, smape, r2):
  r2 = round(r2, 3)
  print(f'MaxE = {maxe}')
  print(f'MSE = {mse}')
  print(f'MAPE = {mape}')
  print(f'SMAPE = {smape}')
  print(f'Coefficient of determination = {r2}')
  # Don't plot >10000 points.
  out_test, out_pred, a = shrink(out_test, out_pred)
  plt.scatter(out_test, out_pred, alpha=a)
  # Force axes to be identical.
  force_axes()
  #plt.title(f'NO\u2082 J rates, R\u00b2={r2}')
  plt.title(f'J rates. R\u00b2={r2}')
  plt.xlabel('J rate from Fast-J / s\u207b\u00b9')
  plt.ylabel('J rate from random forest / s\u207b\u00b9')
  plt.show() 
  plt.close()
 

# Ancilliary functions.

def mem():
  # Memory usage.
  GB = 1000000000
  mem_used = psutil.virtual_memory().used / GB
  print(f'Current memory usage: {mem_used} GB.')
