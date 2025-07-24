'''
Name: Sophie Turner.
Date: 17/7/2024.
Contact: st838@cam.ac.uk
Experiment with NN designs for UKCA data.
'''

import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


GB = 1000000000
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at start of program: {mem} GB.')


# NN design parameters.
# Num layers.
layers_min = 1
layers_max = 10
layers_init = 3
# Num neurons in a layer.
neurons_min = 1
neurons_max = 100000
neurons_init = 20
# Fully connected layers. Adjust later.
connections = 'fc'
# To do: Include nn.Linear alternatives too.

# Transformation function within a neuron. 
trans_fns = ['relu', 'linear', 'bilinear', 
             'threshold', 'hardtanh', 'hardswish', 'relu6', 'elu', 'selu', 'celu',
             'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink',
	     'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 
	     'gumbel_softmax', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu',
	     'mish', 'instance_norm',
             'conv2d', 'conv_transpose2d', 'avg_pool2d', 'max_pool2d', 'max_unpool2d',
             'lp_pool2d', 'adaptive_max_pool2d', 'adaptive_avg_pool2d', 'fractional_max_pool2d',
	     'dropout', 'alpha_dropout', 'feature_alpha_dropout', 'dropout2d', ] 
trans_fn_init = trans_fns[0]
# Learning rate.
lr_min = 0.00001
lr_max = 0.5
lr_init = 0.01
# W&B optimisers.
opts = [torch.optim.Adam]
opt_init = opts[0]
# Max epochs if no convergence.
epochs_max = 500
# Standardisation functions for preprocessing.
stand_fns = [StandardScaler]
stand_fn_init = stand_fns[0] 
# Number input features.
n_inputs_min = 1
n_inputs_max = 16
n_inputs_init = 5
# Feature selection functions.
select_fns = [] 
select_fn_init = None

# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
train_file1 = f'{dir_path}/20170601.npy'
train_file2 = f'{dir_path}/20150115.npy'
model_file = '/scratch/st838/netscratch/tests/nn_model.py' 


# Do feature selection to choose input features.

# Choose parameters for initial NN structure.
n_inputs = n_inputs_init
layers = layers_init
neurons = neurons_init
trans_fn = trans_fn_init

# Write NN module file.
f = open(model_file, 'w')
f.write('import torch')
f.write('\nimport torch.nn as nn')
f.write('\nimport torch.nn.functional as F')

# Write NN class definition.
f.write('\n\nclass Model(nn.Module):')
f.write(f'\n  def __init__(self, inputs={n_inputs}, ')
for i in range(layers):
  f.write(f'h{i+1}={neurons}, ')
f.write('outputs=1):')
f.write('\n    super().__init__()')
f.write('\n    self.fc1 = nn.Linear(inputs, h1)')
if layers > 1:
  for i in range(layers-1):
    f.write(f'\n    self.fc{i+2} = nn.Linear(h{i+1}, h{i+2})')
f.write(f'\n    self.out = nn.Linear(h{layers}, outputs)')
f.write(f'\n  def forward(self, x):')
for i in range(layers):
  f.write(f'\n    x = F.{trans_fn}(self.fc{i+1}(x))')
f.write('\n    x = self.out(x)')
f.write('\n    return(x)')
f.close

# Import the newly written model class.
from nn_model import Model
# Instantiate model.
model = Model()
print(model)
exit()


# Indices of some common combinations to use as inputs and outputs.
phys_all = np.arange(15, dtype=int)
NO2 = 16
HCHO = 19 # Molecular product.
H2O2 = 74
O3 = 78 # O(1D) product.

print('Loading numpy data.')

# Pick a smaller random sample of the data, and do a train-test split.
data = np.load(train_file1)
size = len(data[0])
indices = np.arange(size, dtype=np.int32)
indices = np.random.choice(indices, 10000)
train_idx = np.sort(indices[1000:])
test_idx = np.sort(indices[:1000])
train_data = data[:, train_idx]
test_data = data[:, test_idx]

# Select input and output data.
# Chose the 5 best inputs from linear selection for now. Do proper NN feature selection later.
features = [1,7,8,9,10]
target_idx = HCHO
in_train = train_data[features]
in_test = test_data[features]
out_train = train_data[target_idx]
out_test = test_data[target_idx]

# Make them the right shape.
in_train = np.rot90(in_train, 3)
in_test = np.rot90(in_test, 3)
out_train = out_train.reshape(-1, 1)
out_test = out_test.reshape(-1, 1)

# Standardisation (optional).
scaler = stand_fn_init()
in_train = scaler.fit_transform(in_train)
in_test = scaler.fit_transform(in_test)

# Turn them into torch tensors.
in_train = torch.from_numpy(in_train.copy())
in_test = torch.from_numpy(in_test.copy())
out_train = torch.from_numpy(out_train)
out_test = torch.from_numpy(out_test)

print('\nin_train:', in_train.shape)
print('in_test:', in_test.shape)
print('out_train:', out_train.shape)
print('out_test:', out_test.shape)

# Free up memory.
del(data)
del(train_data)
del(test_data)
del(indices)
del(train_idx)
del(test_idx)

exit()

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
print('\nMSE on test data:', loss)

# Turn them into np arrays for analysis.
out_test, pred = out_test.detach().numpy(), pred.detach().numpy()

# Make them the right shape.
pred = pred.squeeze()
out_test = out_test.squeeze()

print('MAPE:', mean_absolute_percentage_error(out_test, pred))
print('R2:', round(r2_score(out_test, pred), 2))
  
mem = round(psutil.virtual_memory().used / GB, 3)
print(f'Memory usage at end of program: {mem} GB.')  
