'''
Name: Sophie Turner.
Date: 4/6/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a neural network using UKCA data as inputs.
For use on JASMIN's GPUs. 
'''

import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import numpy as np


GB = 1000000000
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at start of program: {mem} GB.')


# Create a class that inherits nn.Module.
class Model(nn.Module):

  # Set up NN structure.
  def __init__(self, inputs=6, h1=8, h2=8, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2) 
    self.out = nn.Linear(h2, outputs) 

  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = self.out(x) 
    return(x)
    

# File paths.
dir_path = '/gws/nopw/j04/um_ml_j_rates'
data_path = f'{dir_path}/data'
results_path = f'{dir_path}/results'
train_files = [f'{data_path}/winter.npy', f'{data_path}/summer.npy']
test_files = [f'{data_path}/20151220.npy', f'{data_path}/20150619.npy', f'{data_path}/20150318.npy', f'{data_path}/20150921.npy'] # 1 day before each training set.
seasons = ['winter', 'summer', 'spring', 'autumn']

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
# Index, full name, simple name.
NO2 = [16, 'NO\u2082', 'NO2']
HCHO = [19, 'HCHO (m)', 'HCHO'] # Molecular product.
H2O2 = [74, 'H\u2082O\u2082', 'H2O2']
O3 = [78, 'O\u2083 \u2192 O(\u00b9D)', 'O3'] # O(1D) product.

# Input features. Chose the best from LR for now. Do proper NN feature selection later.
features = [1,7,8,9,10,14]
# Output target.
target = NO2
target_idx = target[0]
target_name = target[1]
target_simple_name = target[2] # For use in file paths.

for i in range(len(train_files)):
  train_file = train_files[i]
  train_season = seasons[i]
  # Training data.
  train_days = np.load(train_file)
  
  in_train = train_days[features]
  in_train = np.rot90(in_train, 3)
  in_train = torch.from_numpy(in_train.copy())
  out_train = torch.from_numpy(train_days[target_idx])
  out_train = out_train.unsqueeze(1)
  
  # Create instance of model.
  model = Model()

  # Tell the model to measure the error as fitness function to compare pred with label.
  criterion = nn.MSELoss()
  # Choose optimiser and learning rate. Parameters are fc1, fc2, etc. defined above.
  opt = torch.optim.Adam(model.parameters(), lr=0.01)
  
  # Train model.
  epochs = 200 # Choose num epochs.
  print()
  start = time.time()
  for i in range(epochs):
    # Get predicted results.
    pred = model.forward(in_train) 
    # Measure error. Compare predicted values to training targets.
    loss = criterion(pred, out_train)
    # Print every 10 epochs.
    if (i+1) % 10 == 0:
      print(f'Epoch: {i+1}, loss: {loss}')
      mem = round(psutil.virtual_memory().used / GB, 3)
      print(f'Memory usage: {mem} GB.')

    # Backpropagation. Tune weights using loss.
    opt.zero_grad() 
    loss.backward() # Send loss back through the net.
    opt.step() # Step optimiser forward through the net.
  end = time.time()
  minutes = round(((end - start) / 60))
  print(f'Training took {minutes} minutes.')
  
  for j in range(len(test_files)):
    test_file = test_files[j]
    test_season = seasons[j]
    print(f'\nTraining on {train_season} data. Testing with {test_season} data. Target: {target_name}')

    # Testing data.
    test_day = np.load(test_file)

    in_test = test_day[features]
    in_test = np.rot90(in_test, 3)  
    in_test = torch.from_numpy(in_test.copy())
    out_test = torch.from_numpy(test_day[target_idx])
    out_test = out_test.unsqueeze(1)

    print('\nin_train:', in_train.shape)
    print('in_test:', in_test.shape)
    print('out_train:', out_train.shape)
    print('out_test:', out_test.shape)

    # Evaluate model on test set.
    with torch.no_grad(): # Turn off backpropagation.
      pred = model.forward(in_test) # Send the test inputs through the net.
      loss = criterion(pred, out_test) # Compare to test labels.
      r2 = r2_score(out_test.detach().numpy(), pred.detach().numpy())
    print('\nLoss on test data:', loss)
    print('\nR\u00b2:', round(r2, 2))
    
    # Turn them into np arrays for analysis.
    out_test, pred = out_test.detach().numpy(), pred.detach().numpy()
    # View plot.
    title = f'{test_season} {target_name} predictions from NN trained on {train_season} data'
    path = f'{results_path}/{test_season}_{target_simple_name}_by_{train_season}_NN.png'
    fns.show(out_test, out_pred, title, path)

mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at end of program: {mem} GB.') 
