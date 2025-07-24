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
import file_paths as path


GB = 1000000000
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at start of program: {mem} GB.')


# Create a class that inherits nn.Module.
class Model(nn.Module):

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
    

# File paths.
name_file = f'{path.data}/idx_names'
train_file = f'{path.data}/winter.npy'
test_file = f'{path.data}/20150116.npy' 

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
NO2 = 16
HCHO = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.

print('Loading numpy data.')

# Training data.
train_days = np.load(train_file)
# Testing data.
test_day = np.load(test_file)

print('\ntrain data:', train_days.shape)
print('test data:', test_day.shape)

print('\nMaking tensors and selecting inputs and targets.')

# Input features. Chose the 5 best from LR for now. Do proper NN feature selection later.
features = [1,7,8,9,10]

in_train = train_days[features]
in_test = test_day[features]

in_train = np.swapaxes(in_train, 0, 1)
in_test = np.swapaxes(in_test, 0, 1)

in_train = torch.from_numpy(in_train.copy())
in_test = torch.from_numpy(in_test.copy())

# Output target.
target_idx = HCHO
out_train = torch.from_numpy(train_days[target_idx])
out_test = torch.from_numpy(test_day[target_idx])

out_train = out_train.unsqueeze(1)
out_test = out_test.unsqueeze(1)

print('\nin_train:', in_train.shape)
print('in_test:', in_test.shape)
print('out_train:', out_train.shape)
print('out_test:', out_test.shape)

# Create instance of model.
model = Model()

# Tell the model to measure the error as fitness function to compare pred with label.
criterion = nn.MSELoss()
# Choose optimiser and learning rate. Parameters are fc1, fc2, out, defined above.
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

# Evaluate model on test set.
with torch.no_grad(): # Turn off backpropagation.
  pred = model.forward(in_test) # Send the test inputs through the net.
  loss = criterion(pred, out_test) # Compare to test labels.
  r2 = r2_score(out_test.detach().numpy(), pred.detach().numpy())
print('\nLoss on test data:', loss)
print('\nR2:', round(r2, 2))
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at end of program: {mem} GB.') 
