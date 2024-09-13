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

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
  
# ML functions.  
  
def train(in_train, out_train):
  # Set up simple model.
  model = linear_model.LinearRegression()
  model.fit(in_train, out_train)
  return(model)


def test(model, in_test, out_test):
  # Try it out.
  out_pred = model.predict(in_test)
  # See how well it did.
  mse = mean_squared_error(out_test, out_pred)
  mape = mean_absolute_percentage_error(out_test, out_pred)
  r2 = round(r2_score(out_test, out_pred), 3)
  return(out_pred, mse, mape, r2)  
  
  
# Results functions.  

def shrink(out_test, out_pred):
  # Don't plot an unnecessary number of data points i.e. >10000.  
  # Make them the right shape.
  out_test = out_test.squeeze()
  out_pred = out_pred.squeeze()
  length = len(out_pred)
  if length > 10000:
    # Plotting this many datapoints is excessive and costly. Reduce it to 10000.
    idxs = np.int16(np.linspace(0, length, 10000))
    out_test = out_test[idxs]
    out_pred = out_pred[idxs]
    del(idxs)
    # Choose opacity of points.
    alpha = 0.1
  elif length > 100:
    alpha = 0.5
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
  
  
def show(out_test, out_pred, mse, r2):
  r2 = round(r2, 2)
  print(f'MSE = {mse}')
  print(f'Coefficient of determination = {r2}')
  # Don't plot >10000 points.
  out_test, out_pred, a = shrink(out_test, out_pred)
  plt.scatter(out_test, out_pred, alpha=a)
  # Force axes to be identical.
  force_axes()
  #plt.title(f'NO\u2082 J rates, R\u00b2={r2}')
  plt.title(f'H\u2082O\u2082 J rates, R\u00b2={r2}')
  plt.xlabel('J rate from Fast-J / s\u207b\u00b9')
  plt.ylabel('J rate from 5-tree random forest with UKCA physics / s\u207b\u00b9')
  plt.show() 
  plt.close()
