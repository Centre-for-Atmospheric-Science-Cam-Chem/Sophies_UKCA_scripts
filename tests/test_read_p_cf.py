'''
Name: Sophie Turner.
Date: 28/9/2023.
Contact: st838@cam.ac.uk
This is a test to see if the .p file looks right and is read properly using cfPython.
module load anaconda/python3/2022.05
conda activate /home/st838/nethome/condaenv
Alternatively, use pip.
A different version of numpy may be needed for iris. 
Test files are located at scratch/st838/netscratch/scripts/tests.
'''
# Tell this script to run with the currently active Python environment, not the computer's local version. 
#!/usr/bin/env python

import cf

print('Loading test file.')

#test_file = '/scratch/st838/netscratch/tests/atmosa.pd19880901' # Only contains one stash item.
#test_file = '/scratch/st838/netscratch/jHCHO_update/ssp/all_original.pl20150102.pp' # Contains many stash items.
test_file = '/scratch/st838/netscratch/jHCHO_update/ssp/hcho_updated.pn20150102.pp' # 2 items.

O3 = cf.read(test_file,select='stash_code=50502')[0] # J-HCHO

# Investigate the contents.
print('\ntype(O3):')
print(type(O3)) # cf.field.
print('\nO3.shape:')
print(O3.shape) # 23, 85, 144, 192.
print('\nO3:')
print(O3) # A cf field. 
print('\nO3.coords:')
print(O3.coords)
print("\nO3.coordinate('time'):")
print(O3.coordinate('time')) # time(23) days since 2015-1-1 gregorian.
print('\nO3.data:')
print(O3.data) # [[[[1.4139076465625317e-13, ..., 0.0]]]]
print('\nO3.data.shape:')
print(O3.data.shape) # (23, 85, 144, 192)

# See how the indexing works in cfPython.
print('\nO3[4:6]:')
print(O3[4:6]) # A cf field for 2 time points, 6:00 and 7:00.
print("\nO3[0].coordinate('latitude'):")
print(O3[0].coordinate('latitude')) # latitude(144) degrees_north.
print("\nO3[0].coord('latitude'):")
print(O3[0].coord('latitude')) # latitude(144) degrees_north.
print("\nO3[0].coord('latitude').data:")
print(O3[0].coord('latitude').data) # [-89.375, ..., 89.375] degrees_north.
print("\nO3[0].coord('latitude').data[0]:")
print(O3[0].coord('latitude').data[0]) # [-89.375] degrees_north.
print("\nO3[0].coord('atmosphere_hybrid_height_coordinate'):")
print(O3[0].coord('atmosphere_hybrid_height_coordinate')) # atmosphere_hybrid_height_coordinate(85) 1.
print("\nO3[0].coord('atmosphere_hybrid_height_coordinate.a'):")
print(O3[0].coord('atmosphere_hybrid_height_coordinate.a')) 
print("\nO3[0].coord('atmosphere_hybrid_height_coordinate.b'):")
print(O3[0].coord('atmosphere_hybrid_height_coordinate.b')) 
print('\nO3[0,0,0,0]:')
print(O3[0,0,0,0]) # A cf field.
print('\nO3[0,0,0,0].data:')
print(O3[0,0,0,0].data) # [[[[1.4139076465625317e-13]]]].
print('\nO3[0,0,0].data:')
print(O3[0,0,0].data) # [[[[1.4139076465625317e-13, ..., 1.4184003093147685e-13]]]].
print('\nO3[0,0,0].data[0]:')
print(O3[0,0,0].data[0]) # [[[[1.4139076465625317e-13, ..., 1.4184003093147685e-13]]]].

# If .data doesn't work, or you get an error like this: ValueError: PP field data containing 0 words does not match expected length of 27648 words,
# it means the file was not closed properly on creation. Run the model for 1 extra day and don't use the last day.

# Check what we can do with the data.
import numpy as np
print('\nO3.data.flatten():')
print(O3.data.flatten())
print('\nO3.data.flatten().shape:')
print(O3.data.flatten().shape)

# Look at the domain ancillaries.
print('\nO3.domain_ancillaries()')
print(O3.domain_ancillaries())
print('\nO3.domain_ancillaries().items()')
print(O3.domain_ancillaries().items())
print("\nO3.domain_ancillary('domainancillary0")")
print(O3.domain_ancillary('domainancillary0'))
print("\nO3.domain_ancillary('domainancillary0').data")
print(O3.domain_ancillary('domainancillary0').data)

