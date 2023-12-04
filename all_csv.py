'''
Name: Sophie Turner.
Date: 4/12/2023.
Contact: st838@cam.ac.uk
Collect data from daily csvs into one csv.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import glob
import pandas as pd


def merge(in_paths, out_path):
  # Make a DataFrame for the new hourly ATom dataset.
  all_data = pd.DataFrame()
  num_rows = 0
  for each_file in in_paths:
    day_data = pd.read_csv(each_file, index_col=0)
    num_rows += len(day_data)
    all_data = all_data._append(day_data)
  # Sort by date, not by file name.
  all_data = all_data.sort_index()
  # Save it.
  all_data.to_csv(out_path)


# File paths.
dir_path = '/scratch/st838/netscratch/'
UKCA_dir = dir_path + 'nudged_J_outputs_for_ATom/'
ATom_dir = dir_path + 'ATom_MER10_Dataset/'
UKCA_files = glob.glob(UKCA_dir + '/UKCA_hourly_20*.csv') # Just the daily .csv files.
ATom_files = glob.glob(ATom_dir + '/ATom_hourly_20*.csv') 
ATom_out_path = f'{ATom_dir}/ATom_hourly_all.csv'
UKCA_out_path = f'{UKCA_dir}/UKCA_hourly_all.csv'

merge(ATom_files, ATom_out_path)
merge(UKCA_files, UKCA_out_path)







