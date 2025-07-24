'''
Name: Sophie Turner.
Date: 28/11/2024.
Contact: st838@cam.ac.uk.
Make columns of data for ML.
'''

import numpy as np
import matplotlib.pyplot as plt
import file_paths as paths
import constants as con
import prediction_fns as fns

# File paths.
date = '20150621'
day_path = f'{paths.npy}/{date}.npy'

print('Loading data.')
data = np.load(day_path)
print(data.shape)

# Pick some different columns from this day.
# Pick hour, lat, lon for the columns.
points = [con.np12, con.cam12, con.gg12, con.ac12, con.sp12]
for point in points:
  # Pick out the column.
  idx = np.where((data[0] == point[0]) & # Hour.
                 (data[2] == point[1]) & # Lat.
		 (data[3] == point[2]))  # Lon.
  col = data[:, idx].squeeze()
  # Remove upper stratosphere.
  col = col[:, np.where(col[7] > 20)].squeeze()
  # Format date.
  text = fns.format_date(date)  
  # Look at the sunlight flux in the column.
  flux = col[9] + col[10]
  fns.line(flux, col[1], title=f'Column over {point[3]} on {text}', xlab=f'Sunlight flux / {con.Wperm2}', ylab='Altitude / km')
  # Make sure it's not night. Skip this column if it is.
  col = col[:, np.where(flux > 0)].squeeze() 
  print(col.shape)
  if len(col[0]) > 0:
    # Save column.
    col_path = f'{paths.npy}/col{date}_{point[4]}.npy'
    np.save(col_path, col)
