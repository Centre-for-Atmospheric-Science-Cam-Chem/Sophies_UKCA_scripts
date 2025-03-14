'''
Name: Sophie Turner.
Date: 20/11/2023.
Contact: st838@cam.ac.uk
Generate bash scripts to use with Moose for fetching relevant UM output files.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''

import pandas as pd

dir_path = '/scratch/st838/netscratch/'
ATom_dir = 'ATom_MER10_Dataset.20210613/'
ATom_file = dir_path + ATom_dir + 'photolysis_data.csv'

ATom_data = pd.read_csv(ATom_file, index_col=0)
dates = []

for date_time in ATom_data.index:
  date = date_time[0:10]
  date = date.replace('-', '')
  if date not in dates:
    dates.append(date)
    
wait = 14 + ((len(dates) / 5) * (25 - 14))    
    
# Files are too large to move all at once.
# Fetch files from Moose. Run on mass-cli at same time as below.
print('#!/bin/bash')
for i in range(0, len(dates), 5)
  for date in dates[i:i+5]:
    print(f'moo get -v moose:/crum/u-cy731/apl.pp/cy731a.pl{date}.pp .')
  # Wait for transfer. Each transfer takes up to 25 mins.  
  print(f'sleep {wait}m') 
  # Clean up
  print('rm cy*.pp')
  print('rm MetOffice*')        	
	
print('\n\n')

# Move files to netscratch. Run on netscratch at same time as above.
print('#!/bin/bash')
for i in range(0, len(dates), 5):
  # Wait for fetch. Each fetch takes up to 5 mins polling + 9 mins transfer.  
  print('sleep 14m') 
  for date in dates[i:i+5]:
    print(f'scp sophiet@xfer1.jasmin.ac.uk:~/cy731a.pl{date}.pp nudged_J_outputs_for_ATom/')
  # Repeat it to catch files missed by timing mismatch.
  for date in dates[i:i+5]:
    print(f'scp sophiet@xfer1.jasmin.ac.uk:~/cy731a.pl{date}.pp nudged_J_outputs_for_ATom/')


    

