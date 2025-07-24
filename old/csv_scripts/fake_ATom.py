'''
Create some dummy coordinate data to place in the ATom csv style 
for picking out time & space points with all_dates.
'''

import numpy as np
n=100
times = np.random.randint(1, 24, n) # Don't include midnights.
alts = np.random.randint(0, 10000, n) # No altitudes over 10 km.
lats = np.random.rand(n) * 178 - 89
lons = np.random.rand(n) * 360 - 180
for i in range(n):
  '''
  hour = str(times[i])
  if (len(hour) == 1):
    hour = f'0{hour}'
  print(f'01/07/2015 {hour}:00:00')
  '''
  '''
  alt = alts[i]
  print(alt)
  '''
  '''
  lat = lats[i]
  print(lat)
  '''
  '''
  lon = lons[i]
  print(lon)
  '''
