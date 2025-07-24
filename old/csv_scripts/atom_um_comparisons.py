'''
Name: Sophie Turner.
Date: 30/10/2023.
Contact: st838@cam.ac.uk
Script to compare pre-procecced ATom and UKCA data and see differences.
Used on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import glob
import pandas as pd
import comparison_fns_shared as fns
 
# File paths.
path = '/scratch/st838/netscratch/'
out_dir = path + 'analysis'
ATom_dir = path + 'ATom_MER10_Dataset'
UKCA_dir = path + 'nudged_J_outputs_for_ATom'
ATom_file = f'{ATom_dir}/ATom_hourly_all.csv'
UKCA_file = f'{UKCA_dir}/UKCA_hourly_all.csv'
ATom_daily_files = glob.glob(ATom_dir + '/ATom_hourly_20*.csv') 
 
ATom_all = pd.read_csv(ATom_file, index_col=0)
UKCA_all = pd.read_csv(UKCA_file, index_col=0) 

# Look at all the data.
for field in ATom_all.columns:
  ATom_field = ATom_all[field]
  UKCA_field = UKCA_all[field] 
  print(f'\nComparing {field}.')
  fns.diffs(ATom_field, UKCA_field, 'ATom', 'UKCA', out_dir)
  print(f'\nSaving plots for {field} comparisons.\n')
  fns.plot_data(ATom_field, UKCA_field, out_dir, True)
  fns.plot_diff(ATom_field, UKCA_field, out_dir)
  fns.plot_corr(out_dir, ATom_field, UKCA_field, remove_null=True, remove_zero=True)
  
  # Look at each flight.
  for ATom_day_file in ATom_daily_files:
    ATom_day = pd.read_csv(ATom_day_file, index_col=0)
    # There are a lot of flights. Just look at the longest ones.
    if len(ATom_day) >= 10:
      date, _ = fns.split_date_time(ATom_day)
      UKCA_day_file = f'{UKCA_dir}/UKCA_hourly_{date}.csv'
      UKCA_day = pd.read_csv(UKCA_day_file, index_col=0)
      # As long as the data aren't all missing for this flight, plot them.
      if ATom_day[field].notnull().any(): 
        fns.plot_location(ATom_day, UKCA_day, out_dir)
        fns.plot_timeseries(ATom_day[field], UKCA_day[field], out_dir)
        fns.plot_corr(out_dir, ATom_day[field], UKCA_day[field], UKCA_day['LATITUDE'], remove_null=True)    
