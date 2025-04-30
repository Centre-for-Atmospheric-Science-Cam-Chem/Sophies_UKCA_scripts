'''
Name: Sophie Turner.
Date: 30/10/2023.
Contact: st838@cam.ac.uk
Match fields from both ATom and UM pre-processed data.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import pandas as pd

# File paths.
dir_path = '/scratch/st838/netscratch/'
ATom_file = f'{dir_path}tests/ATom_points_2017-10-23.csv'
UKCA_file = f'{dir_path}tests/UKCA_points_2017-10-23.csv'

ATom_data = pd.read_csv(ATom_file, index_col=0)
UKCA_data = pd.read_csv(UKCA_file, index_col=0)

# Find which field items are present in both datasets and discard the rest.
matches = []
for ATom_field in ATom_data.columns:
  for UKCA_field in UKCA_data.columns:
    if ATom_field == UKCA_field:
      matches.append(ATom_field)
   
ATom_data_matched = ATom_data[matches]
UKCA_data_matched = UKCA_data[matches] 

# Make a csv
ATom_out = f'{dir_path}tests/ATom_points_matched.csv'
UKCA_out = f'{dir_path}tests/UKCA_points_matched.csv'
ATom_data_matched.to_csv(ATom_out)
UKCA_data_matched.to_csv(UKCA_out)
