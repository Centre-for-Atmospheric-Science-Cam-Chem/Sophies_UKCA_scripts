'''
Name: Sophie Turner.
Date: 10/6/2025.
Contact: st838@cam.ac.uk
Demonstrate the error reading integer field data in CF Python.
'''

import cf
 
# 1-day .pp data file from the UM.
ukca_file = 'cy731a.pl20150102.pp' 
day = cf.read(ukca_file)

humidity = day[0] # Specific humidity (4d) to show that it works on other fields.
cloud_base = day[1] # Conv cloud base level number (3d) to show that it doesn't work on int fields.
cloud_top = day[2] # Conv cloud base level number (3d) to show that it doesn't work on other int fields..
orography = day[4] # Orography (3d) to show that it works on other 3d fields.

fields = [humidity, orography, cloud_base, cloud_top]
    
# Debug.  
for field in fields:
  print('\nfield.long_name:\n', field.long_name)
  print('\nfield.identity():\n', field.identity())
  print('\nfield:\n', field)
  print('\nfield.dtype:\n', field.dtype)
  print('\nfield[0]:\n', field[0])
  print('\nfield[0].data:\n', field[0].data)
  print('\nfield[0].array:\n', field[0].array)
  print()  
