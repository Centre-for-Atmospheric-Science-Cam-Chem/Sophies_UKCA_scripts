'''
Name: Sophie Turner.
Date: 4/4/2025.
Contact: st838@cam.ac.uk.
Check the order and number of fields.
'''

import cf
import file_paths as paths

# A sample UM output .pp file.
um_file = f'{paths.pp}/cy731a.pl20160601.pp'

# Open it in CF Python.
print('Loading data.')
day = cf.read(um_file)

for field in day:
  print(field.long_name)
  




