'''
Name: Sophie Turner.
Date: 28/9/2023.
Contact: st838@cam.ac.uk
This is a test to see if the .p file looks right and is read properly.
conda activate /home/st838/nethome/condaenv
Alternatively, use pip.
A different version of numpy may be needed for iris. 
Test files are located at scratch/st838/netscratch/tests.
'''
# Tell this script to run with the currently active Python environment, not the computer's local version. 
#!/usr/bin/env python

import iris # pip install scitools-iris

print('Loading test file')

#test_file = './atmosa.pd19880901' # Only contains one stash item.
#test_file = './20150101test.pp' # Contains many stash items.
test_file = './cy731a.pl20150129.pp' # Contains extra physical fields.

#UKCA_data = iris.load_cube(test_file) 
UKCA_data = iris.load_cube(test_file,iris.AttributeConstraint(STASH='m01s00i408')) 
# Investigate the contents.
print('\n\ntype(UKCA_data):')
print(type(UKCA_data)) # iris cube
print('\n\nUKCA_data.shape:')
print(UKCA_data.shape) # 23 t, 85 z, 144 y, 192 x
print('\n\nUKCA_data:')
print(UKCA_data)
print("\n\nUKCA_data.coord('time'):")
print(UKCA_data.coord('time'))
print('\n\nUKCA_data.data:')
print(UKCA_data.data)

# If you get an error like this: ValueError: PP field data containing 0 words does not match expected length of 27648 words,
# it means the file was not closed properly on creation. Run the model for 1 extra day and don't use the last day.






