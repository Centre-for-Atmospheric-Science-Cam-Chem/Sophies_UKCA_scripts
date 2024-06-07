'''
Name: Sophie Turner.
Date: 4/6/2024.
Contact: st838@cam.ac.uk
Try to predict UKCA J rates with a neural network using UKCA data as inputs.
For use on JASMIN's GPUs. 
'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import R2Score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 2 layer
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

'''
# 3 layer
# Create a class that inherits nn.Module.
class Model(nn.Module):

  # Set up NN structure.
  def __init__(self, inputs=5, h1=8, h2=8, h3=8, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)  
    self.out = nn.Linear(h3, outputs) 

  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x)) 
    x = self.out(x) 
    return(x)
'''

# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
name_file = f'{dir_path}/idx_names'
train_file = f'{dir_path}/4days.npy' 

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.linspace(0,13,14, dtype=int)
NO2 = 15
HCHO = 18 # Molecular product.
H2O2 = 73
O3 = 77 # O(1D) product.

print('Loading numpy data.')

# Training data.
days = np.load(train_file)

# Input features. Chose the 5 best from LR for now. Do proper NN feature selection later.
features = [1,7,8,9,10]
inputs = days[features]

# Output target.
target_idx = H2O2
target = days[target_idx]

inputs = np.rot90(inputs, 3)
target = np.reshape(target, (len(target), 1))

in_train, in_test, out_train, out_test = train_test_split(inputs, target, test_size=0.1, random_state=6)

# Standardisation (optional).
scaler = StandardScaler()
in_train = scaler.fit_transform(in_train)
in_test = scaler.fit_transform(in_test)
out_train = scaler.fit_transform(out_train)
out_test = scaler.fit_transform(out_test)

# Make the tensors.
in_train = torch.from_numpy(in_train.copy())
in_test = torch.from_numpy(in_test.copy())
out_train = torch.from_numpy(out_train)
out_test = torch.from_numpy(out_test)

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
epochs = 300 # Choose num epochs.
print()
start = time.time()

for i in range(epochs):
  # Get predicted results.
  pred = model.forward(in_train) 
  # Measure error. Compare predicted values to training targets.
  loss = criterion(pred, out_train)
  # Print every 10 epochs.
  if (i+1) % 10 == 0:
    print(f'Epoch {i+1} \tMSE: {loss.detach().numpy()}')
    
  # Backpropagation. Tune weights using loss.
  opt.zero_grad() 
  loss.backward() # Send loss back through the net.
  opt.step() # Step optimiser forward through the net.
  
end = time.time()
minutes = round(((end - start) / 60))
print(f'Training took {minutes} minutes.')

# Evaluate model on test set.
metric = R2Score()
with torch.no_grad(): # Turn off backpropagation.
  pred = model.forward(in_test) # Send the test inputs through the net.
  loss = criterion(pred, out_test) # Compare to test labels.
  metric.update(pred, out_test)
print('\nLoss on test data:', loss)
print('\nR2 from torcheval:', metric.compute())
print('R2 from sklearn:', round(r2_score(out_test.detach().numpy(), pred.detach().numpy()), 2))

# Remove scaling to view actual values.
out_test = scaler.inverse_transform(out_test.detach().numpy())
pred = scaler.inverse_transform(pred.detach().numpy())

print('out_test after inverse scale:', out_test.shape)
print('pred after inverse scale:', pred.shape)

print('R2 after inverse scale:', round(r2_score(out_test, pred), 2)

# Show a plot of results.
plt.scatter(out_test, pred)
plt.title('jH2O2 predicted by neural network')
plt.xlabel('targets from UKCA')
plt.ylabel('predictions by NN')
plt.show()
