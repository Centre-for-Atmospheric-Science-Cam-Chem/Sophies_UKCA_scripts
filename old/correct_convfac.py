import numpy as np
import pandas as pd


def indices():
  # Random indices to make it faster than looking at all points.
  I = np.random.rand(20)
  t = []
  lon = []
  for i in I:
    t.append(round(i*23))
    lon.append(round(i*191))
  del(I)
  return(t, lon)


path = '/scratch/st838/netscratch/'
path_in = f'{path}nudged_J_outputs_for_ATom/cy731a.pl20180518.pp'
path_npy = f'{path}tests/'
path_out = f'{path_npy}vols.csv'
O3_old = 'stash_code=50228'
O3_added = 'stash_code=50567'

convfac = 1.6605393e-18

data_old = np.load(f'{path_npy}O3_old.npy')
data_added = np.load(f'{path_npy}O3_added.npy')

# Calculate volume and convert.
vol = data_added / (data_old * convfac)
del(data_old)
data_scaled = data_added / (vol * convfac)
del(data_added)  
  
# Pick out mean vols for alt and lat and put them in a 2d array.  
vols = np.zeros((85, 144))
for alt in range(85):
  for lat in range(144):
    zone = np.squeeze(vol[:,alt,lat,:])
    mean = np.nanmean(zone)
    vols[alt, lat] = mean
    
print(vols.shape)

# Put it into a csv for easy viewing.
vols = pd.DataFrame(vols)
print(vols)
vols.to_csv(path_out)
