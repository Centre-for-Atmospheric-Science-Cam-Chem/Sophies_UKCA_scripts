'''
Name: Sophie Turner.
Date: 12/7/2024.
Contact: st838@cam.ac.uk
Compile npy datasets containing multiple days.
For use on JASMIN. 
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import prediction_fns as fns
import psutil

gb = round(psutil.virtual_memory().used / 1000000000, 3) 
print(f'Memory usage at start of script: {gb} GB.')

# Name of compilation.
#name = 'winter'
#name = 'summer'
name = 'spring'
#name = 'autumn'

# File paths.
dir_path = '/gws/nopw/j04/um_ml_j_rates/data'
#input_files = [f'{dir_path}/20151221.npy', f'{dir_path}/20151222.npy', f'{dir_path}/20151223.npy']
#input_files = [f'{dir_path}/20150620.npy', f'{dir_path}/20150621.npy', f'{dir_path}/20150622.npy']
input_files = [f'{dir_path}/20150319.npy', f'{dir_path}/20150320.npy', f'{dir_path}/20150321.npy'] # Spring.
#input_files = [f'{dir_path}/20150922.npy', f'{dir_path}/20150923.npy', f'{dir_path}/20150924.npy']
training_data_path = f'{dir_path}/{name}.npy'

# Get the training data.
days = fns.collate(training_data_path, input_files) 
