import pandas as pd
import numpy as np

path = '/scratch/st838/netscratch/tests/'
path_in = f'{path}vols_interpolated.csv'
convfac = 1.6605393e-18

data_old = np.load(f'{path}O3_old.npy')
data_added = np.load(f'{path}O3_added.npy')
vols = pd.read_csv(path_in)

print('\ndata old:\n', np.min(data_old), np.max(data_old), np.mean(data_old))
print('\ndata added:\n', np.min(data_added), np.max(data_added), np.mean(data_added))

# Use the estimated vols to scale the values.
for alt in range(85):
  for lat in range(144):
    data_added[:,alt,lat,:] = data_added[:,alt,lat,:] / (vols.iloc[alt,lat] * convfac) 

data_added = np.nan_to_num(data_added, nan=0.0, posinf=0.0)

print('\ndata scaled:\n', np.min(data_added), np.max(data_added), np.mean(data_added))
