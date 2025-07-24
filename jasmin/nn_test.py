'''
Tests to see what's slowing down JASMIN's NN training.
Uses Sklearn to allow easier network structure changes.
'''

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPRegressor

warnings.simplefilter('ignore')

layers = [1, 10]
sizes = [1, 10, 100]

print('\nSciKit NN:')

# Num layers.
for nlayers in layers:
  # Num neurons.
  for neurons in sizes:
    # Data size.
    for dsize in sizes:
      datasize = dsize * 1000
      inputs = np.random.normal(size=(datasize, 1))
      targets = np.squeeze(inputs * inputs * 100)
      # Num training epochs.
      for epochs in sizes:
        print(f'\nHidden layers: {nlayers}\nNeurons per layer: {neurons}\nData samples: {inputs.shape}\nTraining epochs: {epochs}') 
        start = time.time()
        if nlayers == 1:
          model = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=epochs, early_stopping=False).fit(inputs, targets)
        elif nlayers == 10:
          model = MLPRegressor(hidden_layer_sizes=(neurons,neurons,neurons,neurons,neurons,neurons,neurons,neurons,neurons,neurons), max_iter=epochs, early_stopping=False).fit(inputs, targets)
        end = time.time()
        seconds = round(end - start, 1)
        print(f'Training time: {seconds} seconds')
     
print ('\nPyTorch NN:')

class Model1(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=1, h1=1, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.out = nn.Linear(h1, outputs) 
  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x))  
    x = self.out(x) 
    return(x)
 
inputs = torch.randn((1000,1))
targets = inputs * inputs * 100
print(f'\nHidden layers: {1}\nNeurons per layer: {1}\nData samples: 1000\nTraining epochs: {1}')
start = time.time()
model = Model1()
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
pred = model.forward(inputs) 
loss = criterion(pred, targets)
opt.zero_grad() 
loss.backward() # Send loss back through the net.
opt.step() # Step optimiser forward through the net. 
end = time.time()
seconds = round(end - start, 1)
print(f'Training time: {seconds} seconds')
    
class Model10(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=1, h1=10, h2=10, h3=10, h4=10, h5=10, h6=10, h7=10, h8=10, h9=10, h10=10, outputs=1):
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
    
inputs = torch.randn((10000, 1))
targets = inputs * inputs * 100
print(f'\nHidden layers: {10}\nNeurons per layer: {10}\nData samples: 10000\nTraining epochs: {10}')
start = time.time()
model = Model10()
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(10):
  pred = model.forward(inputs) 
  loss = criterion(pred, targets)
  opt.zero_grad() 
  loss.backward() # Send loss back through the net.
  opt.step() # Step optimiser forward through the net. 
end = time.time()
seconds = round(end - start, 1)
print(f'Training time: {seconds} seconds')    
    
class Model100(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=1, h1=100, h2=100, h3=100, h4=100, h5=100, h6=100, h7=100, h8=100, h9=100, h10=100, outputs=1):
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
    
inputs = torch.randn((100000, 1))
targets = inputs * inputs * 100
print(f'\nHidden layers: {10}\nNeurons per layer: {100}\nData samples: 100000\nTraining epochs: {100}')
start = time.time()
model = Model100()
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(100):
  pred = model.forward(inputs) 
  loss = criterion(pred, targets)
  opt.zero_grad() 
  loss.backward() # Send loss back through the net.
  opt.step() # Step optimiser forward through the net. 
end = time.time()
seconds = round(end - start, 1)
print(f'Training time: {seconds} seconds') 
