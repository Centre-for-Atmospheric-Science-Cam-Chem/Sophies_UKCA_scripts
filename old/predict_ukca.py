'''
Name: Sophie Turner.
Date: 5/3/2024.
Contact: st838@cam.ac.uk
Experiments with linear regression on UKCA photolysis outputs.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import cf
import glob
import codes_to_names as codes
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Identities of all 68 of the J rates output from Strat-Trop + some physics outputs.
code_names = np.array(codes.code_names)
# Adjust them so that they work with cf identity functions.
for i in range(len(code_names)):
  code = code_names[i,0]
  if code[0:2] == 'UM':
    code = f'id%{code}'
  code_names[i,0] = code
print(code_names, '\n')


def field_group(names):
  # Select a group of fields.
  # names: np array of names of fields.
  group = []
  for name in names:
    print(name)
    idx = np.where(code_names[:,1] == name)
    code = code_names[idx, 0][0][0]
    if code[0:2] == 'UM':
      code = f'id%{code}'
    field = day.select_by_identity(code)
    group.append(field.data)
  return(group)


dir_path = '/scratch/st838/netscratch/nudged_J_outputs_for_ATom/'
pp_files = glob.glob(f'{dir_path}*.pp')

# Physics.
phys_names_all = code_names[:13]
# J rates. 
J_names_all = code_names[13:]

# Indices of main physics.
core_phys_i = [0,1,2,3,4,5,8]
# The main J rates.
core_J_i = [1,3,4,5,9,10,13,14,15,16,17,18,19,21,36,37,38,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
# Summed groups of J rates.
summed_i = [2,6,7,11]
# Make np arrays of their codes.
J_names_core, J_names_summed, phys_names_core = [], [], []
for i in core_J_i:
  J_names_core.append(J_names_all[i])
for i in summed_i:
  J_names_summed.append(J_names_all[i])
for i in core_phys_i:
  phys_names_core.append(code_names[i])
J_names_core, J_names_summed, phys_names_core = np.array(J_names_core), np.array(J_names_summed), np.array(phys_names_core) 

test_file = pp_files[0]
day = cf.read(test_file)
od
# Also get the time and space dims as separate arrays.
field = day[0]
timesteps = field.coord('time').data
alts = field.coord('atmosphere_hybrid_height_coordinate').data
lats = field.coord('latitude').data
lons = field.coord('longitude').data

# Test - select a field.
field_name = 'DOWNWARD SW FLUX'
idx = np.where(code_names[:,1] == field_name)
code = code_names[idx, 0][0][0]
field = day.select_field(code)
piece = field[0,0,72,:]
data = piece.data.squeeze()

# Test - select a field.
field_name = 'jNO2_NO_O3P'
idx = np.where(code_names[:,1] == field_name)
code = code_names[idx, 0][0][0]
field = day.select_field(code)
piece = field[0,0,72,:]
data2 = piece.data.squeeze()

# a 5d array (2 fields, time & space).
in5 = np.array([[[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,0],[0,9]],[[8,7],[6,5]]]],
               [[[[4,3],[2,1]],[[1,1],[2,2]]],[[[3,3],[4,4]],[[5,5],[6,6]]]]]) 
print(in5.ndim)
# a 4d array (1 field, time & space).
out4 = np.array([[[[9,9],[8,8]],[[7,7],[6,6]]],[[[5,5],[4,4]],[[3,3],[2,2]]]])
print(out4.ndim)

inputs = in5
targets = out4
'''
inputs = field_group(phys_names_core[:,1])
targets = field_group(['jNO2_NO_O3P'])
The above inputs are in the wrong shape.
This is how arrays need to be structured:
targets = [2,4,6,8,10]
inputs = np.array([[1,2,3,4,5],[5,4,3,2,1]]) # No.
inputs = np.stack(inputs, axis=1) # Yes.
inputs = [[1,5],[2,4],[3,3],[4,2],[5,1]] # Yes.
'''

# Do linear regression.
print('starting ML with\n', inputs, '\n', targets)
in_train, in_test, out_train, out_test = train_test_split(inputs, targets, test_size=0.2)
model = linear_model.LinearRegression()
#model.fit(in_train.reshape(-1, 1), out_train.reshape(-1, 1))
model.fit(in_train, out_train)
#out_pred = model.predict(in_test.reshape(-1, 1))
out_pred = model.predict(in_test)
#r2 = r2_score(out_test, out_pred)
#r2 = round(r2, 2)
#print(f'Coefficient of determination = {r2}')
plt.scatter(out_test, out_pred)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show() 

# View a plot by longitude.
#piece = np.squeeze(field.data[0,0,72,:])
#plt.plot(lons, piece)
#plt.show()

