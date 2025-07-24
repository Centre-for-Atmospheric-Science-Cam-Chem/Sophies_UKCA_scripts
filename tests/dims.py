import numpy as np


def pad_dim(dim, rep, stride, cols):
  padded = np.empty(0)  
  for i in range(rep):
    new_dim = np.repeat(dim, stride)
    padded = np.append(padded, new_dim)
  cols = np.r_[cols, [padded]] 
  del(new_dim, padded) 
  return(cols)


times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
alts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
lats = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
lons = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

times = [1, 2]
alts = [1, 2, 3]
lats = [1, 2, 3, 4]
lons = [1, 2, 3]

ntimes = len(times)
nalts = len(alts)
nlats = len(lats)
nlons = len(lons)

stride_time = nalts * nlats * nlons 
stride_alt = nlats * nlons 
stride_lat = nlons
stride_lon = 1

rep_time = 1
rep_alt = ntimes
rep_lat = ntimes * nalts
rep_lon = ntimes * nalts * nlats
  
full_size = ntimes * nalts * nlats * nlons  
  
cols = np.empty((0,full_size))   
cols = pad_dim(times, rep_time, stride_time, cols)
cols = pad_dim(alts, rep_alt, stride_alt, cols)
cols = pad_dim(lats, rep_lat, stride_lat, cols)
cols = pad_dim(lons, rep_lon, stride_lon, cols)

print(cols)
print(cols.shape)
