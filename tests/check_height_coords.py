'''
Name: Sophie Turner.
Date: 3/4/2025.
Contact: st838@cam.ac.uk.
Check relationship between hybrid height co-ordinate, model level and altitude in UM output data.
'''

import cf
import file_paths as paths

# A sample UM output .pp file.
um_file = f'{paths.pp}/cy731a.pl20160601.pp'

# Open it in CF Python.
print('Loading data.')
day = cf.read(um_file)
field = day[2]
print(field)

# Fetch a column over the sea and a column over mountains.
# Pacific Ocean: lat = 9, lon = 35 (-145)
# Tibetan Plateau: lat = 30, lon = 268.5 (88.5) 
sea = field[0,:,79,18]
hill = field[0,:,96,143]

print('\nSea:')
print(sea)
print('\nHill:')
print(hill)

# Check the hybrid height coordinate in these. Are they all the same set of numbers? Yes.
print('\nHeights over sea:')
print(sea.coord('atmosphere_hybrid_height_coordinate').data)
print(sea.domain_ancillary('domainancillary0').data) # a
print(sea.domain_ancillary('domainancillary1').data) # b
print('\nHeights over hills:')
print(hill.coord('atmosphere_hybrid_height_coordinate').data)
print(hill.domain_ancillary('domainancillary0').data) # a
print(hill.domain_ancillary('domainancillary1').data) # b
