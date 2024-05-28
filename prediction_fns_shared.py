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

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
  
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
  err = mean_absolute_error(out_test, out_pred)
  #err = round(err, 2)
  r2 = r2_score(out_test, out_pred)
  r2 = round(r2, 2)
  return(out_pred, err, r2)  
  
# Results functions.  
  
def show(out_test, out_pred, mae, r2):
  print(f'Mean squared error = {mae}')
  print(f'Coefficient of determination = {r2}')
  plt.scatter(out_test, out_pred)
  plt.xlabel('actual')
  plt.ylabel('predicted')
  plt.show() 
