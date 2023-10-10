'''
Name: Sophie Turner.
Date: 10/10/2023.
Contact: st838@cam.ac.uk
Pick out relevant data from ATom and UM outputs and put them into simple csv files.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

# Stop unnecessary warnings about pandas version.
import warnings
warnings.simplefilter('ignore')

import cf
import pandas as pd 
import numpy as np

date = '2017-10-23'

# File paths.
UKCA_dir = './StratTrop_nudged_J_outputs_for_ATom/'
ATom_dir = './ATom_MER10_Dataset.20210613/'
UKCA_file = './tests/cy731a.pl20150129.pp' # A test file with extra physical fields. Not for actual use with ATom.
ATom_file = ATom_dir + 'photolysis_data.csv'

# Open the .csv of all the ATom data which I have already pre-processed in the script, ATom_J_data.
print('Loading the ATom data.')
ATom_data = pd.read_csv(ATom_file, index_col=0) # dims = 2. time steps, chemicals + spatial dimensions.

# This step takes several minutes. Better to load individual chunks than the whole thing. See stash codes text file.
print('Loading the UM data.')
UKCA_data = cf.read(UKCA_file)

print('Refining by time comparison.')

# Pick out the fields.
ATom_data = ATom_data[ATom_data.index.str.contains(date)]

# Comparison of hourly time steps.
# ATom and UM are using UTC.
# Get the UM data for the same time of day.
for i in range(len(UKCA_data)):
  field = UKCA_data[i]
  field = field[9:19]
  UKCA_data[i] = field

timesteps = ['2017-10-23 10:00:00', '2017-10-23 11:00:00', '2017-10-23 12:00:00', '2017-10-23 13:00:00', '2017-10-23 14:00:00', 
             '2017-10-23 15:00:00', '2017-10-23 16:00:00', '2017-10-23 17:00:00', '2017-10-23 18:00:00', '2017-10-23 18:59:00']
# Trim so that all arrays are over the same times.
ATom_data = ATom_data.loc[timesteps] # dims= 2. type = pandas dataframe.

print('UKCA_data')
print(type(UKCA_data))
print('\n', UKCA_data, '\n')
print(len(UKCA_data))

#print('Writing csv file')
# Make a .csv.
out_path = f'./tests/ATom_points_{date}.csv'
#ATom_data.to_csv(out_path)
