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
import file_paths as path
import prediction_fns as fns
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


GB = 1000000000
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at start of program: {mem} GB.')


# Create a class that inherits nn.Module.
class SmallModel_5_1(nn.Module):

  # Set up NN structure.
  def __init__(self, inputs=5, h1=8, h2=8, outputs=1):
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


class Model_3_1(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=3, h1=600, h2=600, h3=600, h4=600, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, h4)
    self.out = nn.Linear(h4, outputs) 
  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x))  
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.out(x) 
    return(x)
    

class Model_6_1(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=6, h1=600, h2=600, h3=600, h4=600, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, h4)
    self.out = nn.Linear(h4, outputs) 
  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x))  
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.out(x) 
    return(x)
    

# File paths.
name_file = f'{path.data}/idx_names'
data_file = f'{path.data}/low_res_yr_50.npy'

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
NO2 = 16
HCHO = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.
J_core = [16,18,19,20,24,28,30,31,33,51,52,66,70,71,72,73,74,75,78,79,80,82]

print('Loading numpy data.')

# Training data.
data = np.load(data_file)

# Input features.
features = [1,7,8,9,10,12]
inputs = data[features]

# Output target.
target_idx = NO2
target = data[target_idx]

# Free up some memory.
del(data)

inputs = np.swapaxes(inputs, 0, 1)
target = np.reshape(target, (len(target), 1))

# Standardisation (optional).
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# TTS.
in_train, in_test, out_train, out_test = train_test_split(inputs, target, test_size=0.1, random_state=6, shuffle=False)

# Make the tensors.
in_train = torch.from_numpy(in_train.copy())
in_test = torch.from_numpy(in_test.copy())
out_train = torch.from_numpy(out_train)
out_test = torch.from_numpy(out_test)

print('\nin_train:', in_train.shape)
print('in_test:', in_test.shape)
print('out_train:', out_train.shape)
print('out_test:', out_test.shape)

# Move data to GPU and check GPU.
gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
device = torch.device(gpu)
print(f'\nUsing {gpu}')
in_train = in_train.to(gpu)
out_train = out_train.to(gpu)

# Create instance of model.
model = Model_6_1().to(gpu)

# Tell the model to measure the error as fitness function to compare pred with label.
criterion = nn.MSELoss()
# Choose optimiser and learning rate. Parameters are fc1, fc2, out, defined above.
opt = torch.optim.Adam(model.parameters(), lr=0.005)

# Train model.
epochs = 200 
print()
start = time.time()

for i in range(epochs):
  # Get predicted results.
  pred = model.forward(in_train) 
  # Measure error. Compare predicted values to training targets.
  loss = criterion(pred, out_train)
  # Print every 10 epochs.
  if (i+1) % 20 == 0:
    print(f'Epoch: {i+1}, loss: {loss}')
    
  # Backpropagation. Tune weights using loss.
  opt.zero_grad() 
  loss.backward() # Send loss back through the net.
  opt.step() # Step optimiser forward through the net.

end = time.time()
minutes = round(((end - start) / 60))
print(f'Training took {minutes} minutes.')

# Free up some memory and use it for test set.
del(in_train, out_train)

# Move data to GPU.
in_test = in_test.to(gpu)
out_test = out_test.to(gpu)

# Test model.
with torch.no_grad(): # Turn off backpropagation.
  pred = model.forward(in_test) # Send the test inputs through the net.
  loss = criterion(pred, out_test) # Compare to test labels.

# Turn them into np arrays for analysis.
out_test, pred = out_test.cpu(), pred.cpu()
out_test, pred = out_test.detach().numpy(), pred.detach().numpy()

# Evaluate model on test set.
r2 = r2_score(out_test, pred)
print('\nLoss on test data:', loss)
print('\nR2:', round(r2, 2))
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at end of program: {mem} GB.') 

# View plot.
title = f'NO2 predictions from NN'
path = f'/gws/nopw/j04/um_ml_j_rates/results/lry50_NO2_NN.png'
fns.show(out_test, pred, title, path)
