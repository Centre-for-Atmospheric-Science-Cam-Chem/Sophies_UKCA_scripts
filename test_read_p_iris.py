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
import matplotlib.pyplot as plt
import numpy as np # pip install numpy==1.21
import pandas as pd # pip install --upgrade pandas

print('Loading test file')

#test_file = './atmosa.pd19880901' # Only contains one stash item.
#test_file = './20150101test.pp' # Contains many stash items.
test_file = './cy731a.pl20150101.pp' # Contains extra physical fields.

#UKCA_O3 = iris.load_cube(test_file) 
UKCA_O3 = iris.load_cube(test_file,iris.AttributeConstraint(STASH='m01s50i567')) # O3 --> O(1D)
# Investigate the contents.
print('\n\ntype(UKCA_O3):')
print(type(UKCA_O3)) # iris cube
print('\n\nUKCA_O3.shape:')
print(UKCA_O3.shape) # 23 t, 85 z, 144 y, 192 x
print('\n\nUKCA_O3:')
print(UKCA_O3)
print("\n\nUKCA_O3.coord('time'):")
print(UKCA_O3.coord('time'))
print('\n\nUKCA_O3.data:')
print(UKCA_O3.data)

# If you get an error like this: ValueError: PP field data containing 0 words does not match expected length of 27648 words,
# it means the file was not closed properly on creation. Run the model for 1 extra day and don't use the last day.






