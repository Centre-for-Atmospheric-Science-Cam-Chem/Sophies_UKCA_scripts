'''
Name: Sophie Turner.
Date: 28/9/2023.
Contact: st838@cam.ac.uk
Script to identify which pp fields aren't read properly in cfPython.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import cf

UKCA_file = './cy731a.pl20150129.pp'

# Stash codes. I already know that all the J rate items are OK so don't need to check them.

stash = {'specific humidity':'stash_code=00010',
	 'cloud':'stash_code=00266',
	 'pressure':'stash_code=00408',
	 'zenith angle':'stash_code=01142',
	 'upward sw rad flux':'stash_code=01217', 
	 'downward sw rad flux':'stash_code=01218',
	 'upward lw rad flux':'stash_code=02217',
	 'downward lw rad flux':'stash_code=02218',
	 'temperature':'stash_code=16004',
	 'relative humidity UV grid':'stash_code=30206',
	 'relative humidity T grid':'stash_code=30296', 
	 'heavyside UV grid':'stash_code=30301',
	 'heavyside T grid':'stash_code=30304'}
	
empties = {}	 
for item in stash:
  code = stash.get(item)
  data = cf.read(UKCA_file,select=code)
  if len(data) < 1:
    empties.update({item:code})

print('No data were found for the following items:')
for item in empties:
  print(item, '-', empties.get(item))   
# Shows that all the section 0, 1 and 2 items are empty.
