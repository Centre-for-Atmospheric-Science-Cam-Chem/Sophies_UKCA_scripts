'''
Name: Sophie Turner.
Date: 28/5/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a neural network using UKCA data as inputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/$USER/netscratch_all/st838.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
dir_path = '/scratch/st838/netscratch/ukca_npy'
name_file = f'{dir_path}/idx_names'
# Mini samples of day data for quicker loading and testing.
train_file = f'{dir_path}/train_sample.npy'
test_file = f'{dir_path}/test_sample.npy' 

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.linspace(0,13,14, dtype=int)
NO2 = 15
HCHOm = 18 # Molecular product.
H2O2 = 73
O3 = 77 # O(1D) product.

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

in_train = np.rot90(in_train, 3)
in_test = np.rot90(in_test, 3)

in_train = torch.from_numpy(in_train.copy())
in_test = torch.from_numpy(in_test.copy())

# Output target.
target_idx = NO2
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
for i in range(epochs):
  # Get predicted results.
  pred = model.forward(in_train) 
  # Measure error. Compare predicted values to training targets.
  loss = criterion(pred, out_train)
  # Print every 10 epochs.
  if (i+1) % 10 == 0:
    print(f'Epoch: {i+1}, loss: {loss}')
    
  # Backpropagation. Tune weights using loss.
  opt.zero_grad() 
  loss.backward() # Send loss back through the net.
  opt.step() # Step optimiser forward through the net.

# Evaluate model on test set.
with torch.no_grad(): # Turn off backpropagation.
  out_eval = model.forward(in_test) # Send the test inputs through the net.
  loss = criterion(out_eval, out_test) # Compare to test labels.
print('\nLoss on test data:', loss)
