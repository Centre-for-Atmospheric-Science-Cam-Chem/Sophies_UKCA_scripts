'''
Name: Sophie Turner.
Date: 4/4/2025.
Contact: st838@cam.ac.uk.
Check the order and number of fields in .pp file.
'''

import cf
import file_paths as paths

# A sample UM output .pp file.
um_files = [f'{paths.pp}/cy731a.pl20150103.pp',
            f'{paths.pp}/cy731a.pl20150104.pp',
	    f'{paths.pp}/cy731a.pl20170102.pp']

for um_file in um_files:
  # Open it in CF Python.
  print(f'\n{um_file}')
  
  print('Loading data.')
  day = cf.read(um_file)

  for field in day:
    print(field.long_name)

  print(len(day))
