import glob
import numpy as np

in_files = glob.glob('/scratch/st838/netscratch/ukca_npy/2015*15.npy')
out_file = '/scratch/st838/netscratch/ukca_npy/4days.npy'

days = np.empty((84,0), dtype=np.float32)

for in_file in in_files:
  day = np.load(in_file)
  print(day.shape)
  days = np.hstack((days, day), dtype=np.float32)
  print(days.shape)
  print(type(days[0,0]))

np.save(out_file, days)
