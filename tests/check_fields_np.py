'''
Name: Sophie Turner.
Date: 4/4/2025.
Contact: st838@cam.ac.uk.
Check the contents and number of fields in .npy file.
'''

import numpy as np
import file_paths as paths

# A sample UM output .pp file.
npy_file = f'{paths.npy}/test_day.npy'

# Open it in CF Python.
print('Loading data.')
data = np.load(npy_file)
print(data.shape) 

# Check range of each field.
for i in range(len(data)):
  field = data[i]
  print(f'field {i}: {min(field)} to {max(field)}')  
