'''
Name: Sophie Turner.
Date: 20/11/2023.
Contact: st838@cam.ac.uk
Generate commands to use with Moose for fetching relevant UM output files.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''

import pandas as pd

dir_path = '/scratch/st838/netscratch/'
ATom_dir = dir_path + 'ATom_MER10_Dataset/'
ATom_file = ATom_dir + 'photolysis_data.csv'
UKCA_dir = dir_path + 'nudged_J_outputs_for_ATom/'

ATom_data = pd.read_csv(ATom_file, index_col=0)
dates = []

for date_time in ATom_data.index:
  date = date_time[0:10]
  date = date.replace('-', '')
  if date not in dates:
    dates.append(date)  
    
# Files are too large to move all at once.
# Fetch files from Moose. Run on mass-cli at same time as below.
for i in range(0, len(dates), 6): 
  # Clean up
  print('rm cy*.pp')
  print('rm MetOffice*') 
  for date in dates[i:i+6]:
    print(f'moo get -v moose:/crum/u-cy731/apl.pp/cy731a.pl{date}.pp .')
  print()
# Clean up
print('rm cy*.pp')
print('rm MetOffice*')    	
	
print('\n\n')

# Pull files to netscratch. Run on netscratch at same time as above.
for i in range(0, len(dates), 6):
  print('sleep 16m')
  for date in dates[i:i+6]:
    print(f'scp sophiet@xfer1.jasmin.ac.uk:~/cy731a.pl{date}.pp {UKCA_dir}')
  print()


    

