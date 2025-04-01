'''
Name: Sophie Turner.
Date: 28/9/2023.
Contact: st838@cam.ac.uk
This is a test to see if the .p file looks right and is read properly using cfPython.
module load anaconda/python3/2022.05
conda activate /home/st838/nethome/condaenv
Alternatively, use pip.
A different version of numpy may be needed for iris. 
Test files are located at scratch/st838/netscratch/tests.
'''
# Tell this script to run with the currently active Python environment, not the computer's local version. 
#!/usr/bin/env python

import cf

print('Loading test file.')

#test_file = './atmosa.pd19880901' # Only contains one stash item.
#test_file = './20150101test.pp' # Contains many stash items.
test_file = './cy731a.pl20150101.pp' # Contains extra physical fields.

# Investigate the contents.
print('\ntype(UKCA_O3):')
print(type(UKCA_O3)) # cf.field.
print('\nUKCA_O3.shape:')
print(UKCA_O3.shape) # 23, 85, 144, 192.
print('\nUKCA_O3:')
print(UKCA_O3) # A cf field. 
print("\nUKCA_O3.coordinate('time'):")
print(UKCA_O3.coordinate('time')) # time(23) days since 2015-1-1 gregorian.
print('\nUKCA_O3.data:')
print(UKCA_O3.data) # [[[[1.4139076465625317e-13, ..., 0.0]]]]

# See how the indexing works in cfPython.
print('\nUKCA_O3[4:6]:')
print(UKCA_O3[4:6]) # A cf field for 2 time points, 6:00 and 7:00.
print("\nUKCA_O3[0].coordinate('latitude'):")
print(UKCA_O3[0].coordinate('latitude')) # latitude(144) degrees_north.
print("\nUKCA_O3[0].coord('latitude'):")
print(UKCA_O3[0].coord('latitude')) # latitude(144) degrees_north.
print("\nUKCA_O3[0].coord('latitude').data:")
print(UKCA_O3[0].coord('latitude').data) # [-89.375, ..., 89.375] degrees_north.
print("\nUKCA_O3[0].coord('latitude').data[0]:")
print(UKCA_O3[0].coord('latitude').data[0]) # [-89.375] degrees_north.
print("\nUKCA_O3[0].coord('atmosphere_hybrid_height_coordinate'):")
print(UKCA_O3[0].coord('atmosphere_hybrid_height_coordinate')) # atmosphere_hybrid_height_coordinate(85) 1.
print('\nUKCA_O3[0,0,0,0]:')
print(UKCA_O3[0,0,0,0]) # A cf field.
print('\nUKCA_O3[0,0,0,0].data:')
print(UKCA_O3[0,0,0,0].data) # [[[[1.4139076465625317e-13]]]].
print('\nUKCA_O3[0,0,0].data:')
print(UKCA_O3[0,0,0].data) # [[[[1.4139076465625317e-13, ..., 1.4184003093147685e-13]]]].
print('\nUKCA_O3[0,0,0].data[0]:')
print(UKCA_O3[0,0,0].data[0]) # [[[[1.4139076465625317e-13, ..., 1.4184003093147685e-13]]]].

# If you get an error like this: ValueError: PP field data containing 0 words does not match expected length of 27648 words,
# it means the file was not closed properly on creation. Run the model for 1 extra day and don't use the last day.






