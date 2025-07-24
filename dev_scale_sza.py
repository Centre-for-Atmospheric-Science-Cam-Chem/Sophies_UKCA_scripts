'''
Name: Sophie Turner.
Date: 28/3/2025.
Contact: st838@cam.ac.uk.
Test scaling solar zenith angle with vertical height.
'''

import numpy as np
import constants as con
import file_paths as paths


def scale_sza_by_alt(sza_sea, alt):
  '''
  Scale solar zenith angle with vertical height based on atmospheric pressure, at a specific UM grid point. 
  
  Parameters:
  sza_sea (float): surface-level solar zenith angle in cos radians at one UM grid point. 
  pres_sea (float): surface-level air pressure in Pa for the vertical column that this grid point is in.
  pres_point (float): air pressure in Pa at this grid point.
  alt (float): altitude in metres at this point.
  
  Returns:
  sza (float): scaled SZA in cos radians at this grid point.
  '''

  # Calculate the scaled SZA using the pressure ratio.
  #sza = math.acos(math.cos(sza_sea) * (pres_sea / pres_point))
  
  # Earth's radius (m).
  rad = 6.371e6
  # Try to replicate the way Fast-J scales SZA to get similar retsuls.
  sza = np.sqrt(1 - ((rad / (rad + alt))**2) * (1 - sza_sea**2))
 
  return(np.float32(sza))

'''
# TEST: Differences. The SZA should be unchanged at surface level, and if the sun is directly overhead.
sza = 45
alt = 5000
pres_sea = 100000
pres_point = 50000
print('sza before:', sza)
sza = scale_sza_by_pressure(sza, pres_sea, pres_point, alt)
print('sza after:', sza)
'''

# Load training dataset.
print('\nLoading data.')
filename = 'low_res_yr_50'
data = np.load(f'{paths.npy}/{filename}.npy') # (85, 182000000)
print(data.shape)

'''
# For each of the samples...
for i in range(len(data[0])):  
  
  # Check.
  print(data[con.sza, i])
  
  point = data[:, i] # (85, )
    
  # Get the sea-level SZA. 
  sza = point[con.sza]

  # Get the altitude.
  alt = point[con.alt]
  
  # Convert altitude to metres.
  metres = alt * 85 * 1000 # 85 levels, km to m.
  
  
  # Get the pressure.
  pres = point[con.pressure] 
  
  # Get the column this point is in.
  lat = point[con.lat]
  lon = point[con.lon]
  time = point[con.days]
  col = data[:, (data[con.days] == time) & (data[con.lat] == lat) & (data[con.lon] == lon)].squeeze()
  
  # TEST
  print('\ncol:')
  print(col)
  print(type(col))
  print(col.shape)
  exit()
  
  # TEST: Is the highest pressure in a column always at the lowest altitude?
  # Find the index of the highest pressure in this column.
  idx_col_max_pres = col[np.argmax(col[con.pres])]
  print(idx_col_mas_pres)
  alt_col_max_pres = col[con.alt, idx_col_max_pres]
  print(alt_col_max_pres)
  exit()
 
  # Get the sea-level pressure in this column.
  pres_sea = col[np.argmax(col[con.pres])]

  # TEST: Is the sea-level pressure the same everywhere?
  print(pres_sea)
  
  # Adjust the SZA.
  sza = scale_sza_by_alt(sza, metres)  

  # Replace the SZA value in the dataset.
  data[con.sza, i] = sza
'''   

print('Scaling solaz zenith angles.') 
szas = data[con.sza]
alts = data[con.alt]  
metres = alts * 85000
szas = scale_sza_by_alt(szas, metres) 
data[con.sza] = szas 


# Save the new training dataset.
out_path = f'{paths.npy}/{filename}_scaled_sza.npy'
print(f'Saving new dataset at {out_path}.')
np.save(filename, data)
