import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPRegressor


class Model2(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=1, h1=100, h2=100, outputs=1):
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


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
device = torch.device(dev)
print(f'Using {dev}')
inputs = torch.randn((100000, 1))
targets = inputs * inputs * 100
inputs = inputs.to(device)
targets = targets.to(device)
print(f'Targets: {targets}')
print(f'\nHidden layers: {2}\nNeurons per layer: {100}\nData samples: 100000\nTraining epochs: {100}')
start = time.time()
model = Model2().to(device)
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
