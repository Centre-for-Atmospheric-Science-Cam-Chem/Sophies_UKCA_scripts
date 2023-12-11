import numpy as np

path = '/scratch/st838/netscratch/'
path_in = f'{path}nudged_J_outputs_for_ATom/cy731a.pl20180518.pp'
path_npy = f'{path}tests/'
O3_old = 'stash_code=50228'
O3_added = 'stash_code=50567'
convfac = 1.6605393e-18

data_old = np.load(f'{path_npy}O3_old.npy')
data_added = np.load(f'{path_npy}O3_added.npy')

# Calculate volume and convert.
vol = data_added / (data_old * convfac)
data_scaled = data_added / (vol * convfac)

# Pick out mean vols for alt and lat and put them in a 2d array.
I = np.random.rand(20)
t = []
lon = []
for i in I:
  t.append(round(i*23))
  lon.append(round(i*191))
  
vols = np.zeros((85, 144))
for alt in range(85):
  for lat in range(144):
    zone = np.squeeze(vol[t,alt,lat,lon])
    mean = np.nanmean(zone)
    vols[alt, lat] = mean
    print(mean)
    
print(vols)
print(vols.shape)
