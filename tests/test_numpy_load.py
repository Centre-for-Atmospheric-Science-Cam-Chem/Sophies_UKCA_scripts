import numpy as np


path = '/scratch/st838/netscratch/jHCHO_update/ssp/'
npfile_1 = f'{path}numpy_data_original.npy'
npfile_2 = f'{path}numpy_data_updated.npy'
data_1 = np.load(npfile_1)
data_2 = np.load(npfile_2)

print(data_1.shape)
print(data_2.shape)

data_1 = np.squeeze(data_1)
data_2 = np.squeeze(data_2)

print(data_1.shape)
print(data_2.shape)
