'''
Name: Sophie Turner.
Date: 28/6/2024.
Contact: st838@cam.ac.uk
Make a .npy file containing data from multiple days of .npy daily files. 
Files are located at scratch/$USER/netscratch_all/st838.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import psutil
import prediction_fns_numpy as fns

start_mem = psutil.virtual_memory().used
gb = round(start_mem / 1000000000, 3)
print(f'Memory usage is {gb} GB at start of script.')

# File paths.
dir_path = '/scratch/st838/netscratch/ukca_npy'
input_files = [f'{dir_path}/20150115.npy', f'{dir_path}/20150415.npy', f'{dir_path}/20150715.npy', f'{dir_path}/20151015.npy']
training_data_path = f'{dir_path}/4days_test.npy'

# Get the training data.
days = fns.collate(training_data_path, input_files) 

end_mem = psutil.virtual_memory().used
gb = round(end_mem / 1000000000, 3)
print(f'Memory usage is {gb} GB at end of script.')
