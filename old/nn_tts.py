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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 2 layer.
# Create a class that inherits nn.Module.
class SmallModel(nn.Module):

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


# 12 layer.
# Create a class that inherits nn.Module.
class BigModel(nn.Module):

  # Set up NN structure.
  def __init__(self, inputs=5, h1=20, h2=20, h3=20, h4=20, h5=20, h6=10, h7=10, h8=10, h9=10, h10=10, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, h4)
    self.fc5 = nn.Linear(h4, h5)
    self.fc6 = nn.Linear(h5, h6)
    self.fc7 = nn.Linear(h6, h7)
    self.fc8 = nn.Linear(h7, h8)
    self.fc9 = nn.Linear(h8, h9)
    self.fc10 = nn.Linear(h9, h10)
    self.out = nn.Linear(h10, outputs) 

  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = F.relu(self.fc3(x)) 
    x = F.relu(self.fc4(x)) 
    x = F.relu(self.fc5(x)) 
    x = F.relu(self.fc6(x)) 
    x = F.relu(self.fc7(x)) 
    x = F.relu(self.fc8(x)) 
    x = F.relu(self.fc9(x)) 
    x = F.relu(self.fc10(x)) 
    x = self.out(x) 
    return(x)


# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
name_file = f'{dir_path}/idx_names'
train_file = f'{dir_path}/4days.npy' 

# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
J_all = np.arange(15, 85, dtype=int)
NO2 = 16
HCHO = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.

print('Loading numpy data.')

# Training data.
days = np.load(train_file)

# Input features. Chose the 5 best from LR for now. Do proper NN feature selection later.
features = [1,7,8,9,10]
inputs = days[features]

# Output target.
target_idx = H2O2
target = days[target_idx]

print('inputs',inputs)

inputs = np.swapaxes(inputs, 0, 1)
target = np.reshape(target, (len(target), 1))
#inputs = inputs.reshape(-1, 1)
#target = target.reshape(-1, 1)

in_train, in_test, out_train, out_test = train_test_split(inputs, target, test_size=0.1, random_state=6, shuffle=False)

# Standardisation (optional).
scaler = StandardScaler()
in_train = scaler.fit_transform(in_train)
in_test = scaler.fit_transform(in_test)

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
model = SmallModel()

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
with torch.no_grad(): # Turn off backpropagation.
  pred = model.forward(in_test) # Send the test inputs through the net.
  loss = criterion(pred, out_test) # Compare to test labels.
print('\nMSE on test data:', loss)

# Turn them into np arrays for analysis.
out_test, pred = out_test.detach().numpy(), pred.detach().numpy()

# Make them the right shape.
pred = pred.squeeze()
out_test = out_test.squeeze()

print('MAPE:', mean_absolute_percentage_error(out_test, pred))
print('R2:', round(r2_score(out_test, pred), 2))
'''
# Plotting this many datapoints is excessive and costly. 
length = len(pred)
# Reduce it to 10%.
idxs = np.arange(0, length, 10)
pred = pred[idxs]
out_test = out_test[idxs]
del(idxs)

# Show a plot of results.
plt.figure()
plt.scatter(out_test, pred, alpha=0.1)
plt.title(name)
plt.xlabel('targets from UKCA')
plt.ylabel('predictions by NN')
plt.savefig(f'{dir_path}/{name}.png')
plt.close()
'''
