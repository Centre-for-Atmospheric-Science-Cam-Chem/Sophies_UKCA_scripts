'''
Name: Sophie Turner.
Date: 4/4/2025.
Contact: st838@cam.ac.uk.
Check the order and number of fields in .pp file and their values' ranges.
'''

import cf
import file_paths as paths

# A sample UM output .pp file.
um_files = [f'{paths.pp}/atmosa.pl19810901_18']

for um_file in um_files:
  # Open it in CF Python.
  print(f'\n{um_file}')
  
  print('Loading data.')
  day = cf.read(um_file)

  for field in day:
    print(f'{field.long_name} ranges from {field.min()} to {field.max()}')

  print(f'There are {len(day)} fields in the dataset.')
