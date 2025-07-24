import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

mem_start = psutil.virtual_memory().used


class Model1(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=5, h1=10, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.out = nn.Linear(h1, outputs) 
  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x))  
    x = self.out(x) 
    return(x)
    
class Model2(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=5, h1=10, h2=10, outputs=1):
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
    
class Model3(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=5, h1=10, h2=10, h3=10, outputs=1):
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
    
class Model4(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=5, h1=10, h2=10, h3=10, h4=10, outputs=1):
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
    
class Model5(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=5, h1=10, h2=10, h3=10, h4=10, h5=10, outputs=1):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, h4)
    self.fc5 = nn.Linear(h4, h5)
    self.out = nn.Linear(h5, outputs) 
  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x))  
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = self.out(x) 
    return(x)
    
class Model10(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=5, h1=10, h2=10, h3=10, h4=10, h5=10, h6=10, h7=10, h8=10, h9=10, h10=10, outputs=1):
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

class Model20(nn.Module):
  # Set up NN structure.
  def __init__(self, inputs=5, h1=10, h2=10, h3=10, h4=10, h5=10, h6=10, h7=10, h8=10, h9=10, h10=10,
               h11=10, h12=10, h13=10, h14=10, h15=10, h16=10, h17=10, h18=10, h19=10, h20=10, outputs=1):
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
    self.fc11 = nn.Linear(h10, h11) 
    self.fc12 = nn.Linear(h11, h12)
    self.fc13 = nn.Linear(h12, h13)
    self.fc14 = nn.Linear(h13, h14)
    self.fc15 = nn.Linear(h14, h15)
    self.fc16 = nn.Linear(h15, h16)
    self.fc17 = nn.Linear(h16, h17)
    self.fc18 = nn.Linear(h17, h18)
    self.fc19 = nn.Linear(h18, h19)
    self.fc20 = nn.Linear(h19, h20)
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
    x = F.relu(self.fc11(x))  
    x = F.relu(self.fc12(x))
    x = F.relu(self.fc13(x))
    x = F.relu(self.fc14(x))
    x = F.relu(self.fc15(x))
    x = F.relu(self.fc16(x))
    x = F.relu(self.fc17(x))
    x = F.relu(self.fc18(x))
    x = F.relu(self.fc19(x))
    x = F.relu(self.fc20(x))
    x = self.out(x) 
    return(x)


# File paths.
dir_path = 'data'
train_file = f'{dir_path}/4days.npy'
test_file = f'{dir_path}/20150601.npy' 
NO2 = 16
GB = 1000000000

start_load = time.time()

# Fresh data load.
train_days = np.load(train_file)
test_day = np.load(test_file)

end_load = time.time()
m_load = round((end_load - start_load) / 60, 1)
print(f'Fresh data loading took {m_load} minutes.')

start_load = time.time()

# Cached data load.
train_days = np.load(train_file)
test_day = np.load(test_file)

end_load = time.time()
m_load = round((end_load - start_load) / 60, 1)
print(f'Cached data loading took {m_load} minutes.')

print('\ntrain data:', train_days.shape)
print('test data:', test_day.shape)

# Select input and output data.
# Chose the 5 best inputs from linear selection for now. Do proper NN feature selection later.
features = [1,7,8,9,10]
target_idx = NO2
in_train = train_days[features]
in_test = test_day[features]
out_train = train_days[target_idx]
out_test = test_day[target_idx]

start_pre = time.time()

# Make them the right shape.
in_train = np.swapaxes(in_train, 0, 1)
in_test = np.swapaxes(in_test, 0, 1)
out_train = out_train.reshape(-1, 1)
out_test = out_test.reshape(-1, 1)

# Standardisation (optional).
scaler = StandardScaler()
in_train = scaler.fit_transform(in_train)
in_test = scaler.fit_transform(in_test)

# Turn them into torch tensors.
in_train = torch.from_numpy(in_train.copy())
in_test = torch.from_numpy(in_test.copy())
out_train = torch.from_numpy(out_train)
out_test = torch.from_numpy(out_test)

# Free up memory.
del(train_days)
del(test_day)

end_pre = time.time()
s_pre = round(end_pre - start_pre)
print(f'Preprocessing took {s_pre} seconds.')

models = [Model1, Model2, Model3, Model4, Model5, Model10, Model20]
epochs = 10

for m in range(len(models)):
  Model = models[m]
  n = m + 1
  if n == 6:
    n = 10
  elif n == 7:
    n = 20
  print(f'\n{n}-layer NN:')
  
  start_train = time.time()
  
  # Create instance of model.
  model = Model()
  # Tell the model to measure the error as fitness function to compare pred with label.
  criterion = nn.MSELoss()
  # Choose optimiser and learning rate. Parameters are fc1, fc2, out, defined above.
  opt = torch.optim.Adam(model.parameters(), lr=0.01)

  # Train model.
  for i in range(epochs):
    # Get predicted results.
    pred = model.forward(in_train) 
    # Measure error. Compare predicted values to training targets.
    loss = criterion(pred, out_train)
    mem_end1 = psutil.virtual_memory().used 
    if mem_end1 < mem_start:
      mem_start = 0
    # Backpropagation. Tune weights using loss.
    opt.zero_grad() 
    loss.backward() # Send loss back through the net.
    opt.step() # Step optimiser forward through the net.
  
  end_train = time.time()
  m_train = round((end_train - start_train) / 60, 1)
  print(f'Training took {m_train} minutes.')

  start_test = time.time()
  
  # Evaluate model on test set.
  with torch.no_grad(): # Turn off backpropagation.
    pred = model.forward(in_test) # Send the test inputs through the net.
    loss = criterion(pred, out_test) # Compare to test labels.
    mem_end2 = psutil.virtual_memory().used 
    
  end_test = time.time()
  s_test = round((end_test - start_test), 1)
  print(f'Testing the model took {s_test} seconds.')
  
  mem_end = max([mem_end1, mem_end2])
  mem_total = round((mem_end - mem_start) / GB, 1)
  print(f'Maximum memory usage at one time: {mem_total} GB.')
