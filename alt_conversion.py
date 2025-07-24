'''
Name: Sophie Turner.
Date: 13/5/2025.
Contact: st838@cam.ac.uk.
'''

import time
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

print('\nWith the correct altitudes.')
data_file = f'{paths.npy}/test_day.npy'
data = np.load(data_file) 

# Get the unique hour values.
hours = np.unique(data[con.hour])
# Pick out 1 timestep because the orography doesn't change in time.
ts = data[:, data[con.hour] == hours[0]] # First timestep.

# Get the unique lat & lon values.
lats = np.unique(ts[con.lat])
lons = np.unique(ts[con.lon])

# Altitude level.
level = data[con.alt].copy()

start = time.time()

# Loop through every vertical column, calculating altitude.
# For each lat...
for i in range(len(lats)):
  lat = lats[i]  
  print(f'Processing latitude {i + 1} of {len(lats)}')
  # And each lon...
  for j in range(len(lons)):
    lon = lons[j]
    # Get the indices of all the data in this column.
    idx = np.where((ts[con.lat] == lat) & (ts[con.lon] == lon))[0]
    # Ground height.
    ground = ts[7, idx[0]]
    # Calculate the altitudes of levels in this column.
    alts = fns.lvl_to_alt(ground, True)
    # Repeat the altitudes x the number of timesteps.
    alts = np.tile(alts, len(hours))
    # Find this column for every timestep.
    idx = np.where((data[con.lat] == lat) & (data[con.lon] == lon))[0]
    # Replace the column of altitudes for every timestep.
    data[con.alt, idx] = alts
  
end = time.time()
minutes = (end - start) / 60
print(f'It took {minutes} minutes to calculate all the altitudes.') 

# Save the updated dataset.
np.save(f'{paths.npy}/test_day_corrected_alt.npy', data)

# Check that they are different.
plt.scatter(level, data[con.alt])
plt.xlabel('Altitude before orography calculation')
plt.ylabel('Altitude after orography calculation')
plt.show()

input_i = [0,1,2,3,4,5,8,9,10,11,12,16] # Original.
target_i = np.arange(18, 88)
# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, input_i, target_i)

# 90/10 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets)

# Make the regression model.
model = RandomForestRegressor(n_estimators=20, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)
model.fit(in_train, out_train)
out_pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)

# View performance.
fns.show(out_test, out_pred, maxe, mse, mape, smape, r2)
