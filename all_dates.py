'''
Name: Sophie Turner.
Date: 3/11/2023.
Contact: st838@cam.ac.uk
Compile data from daily UM .pp files.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import glob
import pandas as pd

# File paths.
dir_path = '/scratch/st838/netscratch/'
UKCA_dir = dir_path + 'nudged_J_outputs_for_ATom/'
ATom_dir = dir_path + 'ATom_MER10_Dataset.20210613/'
ATom_file = ATom_dir + 'photolysis_data.csv'


def get_date(UKCA_file):
  # Find dates of UKCA files.
  year = UKCA_file[-11:-7]
  month = UKCA_file[-7:-5]
  day = UKCA_file[-5:-3]
  return(year, month, day)


def get_ATom_day(UKCA_file):
  # Pick out the corresponding date from ATom.
  year, month, day = get_date(UKCA_file)
  date_str = f'{year}-{month}-{day}'
  ATom_day = ATom_data[ATom_data.index.str.contains(date_str)]
  return(ATom_day)


def get_times(ATom_day):
  # Find out what times of day we have data for.
  ATom_hours = ATom_day[ATom_day.index.str.endswith('00:00')]
  # Apply a buffer so that we don't chop off start and end times which are close to an extra hour.
  if int(ATom_day.index[0][-8:-6]) < int(ATom_hours.index[0][-8:-6]): # start hours.
    if int(ATom_day.index[0][-5:-3]) <= 10: # 10 minute buffer for start minutes.
      # Add this extra time point as 1st item.
      extra = ATom_day.iloc[0:1] 
      ATom_hours = pd.concat([extra, ATom_hours])
  if int(ATom_day.index[-1][-8:-6]) != int(ATom_hours.index[-1][-8:-6]): # end hours.
    if int(ATom_day.index[-1][-5:-3]) >= 50: # 10 minute buffer for end minutes.
      # Add this as last item.  
      extra = ATom_day.iloc[-1] 
      ATom_hours = ATom_hours._append(extra)
  

# Open ATom dataset, already partially pre-processed.
ATom_data = pd.read_csv(ATom_file, index_col=0)

# Find dates of UKCA files.
all_files = glob.glob(UKCA_dir + '/*.pp') # Just .pp files.
for each_file in all_files:
  # Pick out the corresponding date from ATom.
  ATom_day = get_ATom_day(each_file)
  if not ATom_day.empty:
    get_times(ATom_day)
 
