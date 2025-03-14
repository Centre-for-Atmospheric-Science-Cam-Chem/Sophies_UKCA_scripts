'''
Name: Sophie Turner.
Date: 16/1/2023.
Contact: st838@cam.ac.uk
Try to predict J rates using ML with the ATom data.
For use on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import prediction_fns_shared as fns
import pandas as pd

# File paths.
dir_path = '/scratch/st838/netscratch/'
ATom_dir = dir_path + 'ATom_MER10_Dataset/'
ATom_file = ATom_dir + 'photolysis_data.csv'

# Open ATom dataset, already partially pre-processed.
ATom_data = pd.read_csv(ATom_file)
ATom_data = ATom_data.rename(columns={'UTC_Start_dt':'TIME', 'T':'TEMPERATURE', 'G_LAT':'LATITUDE', 
                                      'G_LONG':'LONGITUDE', 'G_ALT':'ALTITUDE', 'Pres':'PRESSURE',
				      'cloudindicator_CAPS':'CLOUD %'})    
ATom_data = fns.standard_names(ATom_data)  

# Get time info as float.
ATom_data = fns.time_to_s(ATom_data)

# Set up data structures for different experiments.
not_used = ['TIME', 'CLOUDFLAG AMS', 'JO3 DNWFRAC', 'JNO2 DNWFRAC']
ATom_data = ATom_data.drop(columns=not_used)
J_all = ATom_data.loc[:,ATom_data.columns.str.startswith('J')].columns.tolist()
phys_all = ['SECONDS', 'TEMPERATURE', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 
            'PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY']			  
phys_reduced = ['TEMPERATURE', 'PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY']
phys_min = ['PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE']

# Test using each J as the only input.
best, worst = [], []
for J in J_all:
  input_name = [J]
  target_name = J_all.copy()
  target_name, input_name, ATom_data = prep_data(target_name, input_name, ATom_data, J_all)
  in_train, in_test, out_train, out_test = fns.set_params(target_name, input_name, ATom_data, ATom_data)
  model = fns.train(in_train, out_train)
  out_pred, mse, r2 = fns.test(model, in_test, out_test)
  if r2 >= 0.88:
    best.append([input_name[0], r2])
    fns.show(out_test, out_pred, mse, r2)
  elif r2 <= 0.6:
    worst.append([input_name[0], r2])

# Look at the results.
best.sort(reverse = True, key = lambda x: x[1])
worst.sort(key = lambda x: x[1])
print('\nbest:')
for J in best:
  print(J)
print('worst:')
for J in worst:
  print(J)   
# The best from the above were JH2O2 OH OH, JCH3OOH CH3O OH and JCHOCHO CH2O CO with r2 0.89.

input_name = phys_all
target_name = ['JNO2 NO O3P']
target_name, input_name, ATom_data = fns.prep_data(target_name, input_name, ATom_data, J_all)
in_train, in_test, out_train, out_test = fns.set_params(target_name, input_name, ATom_data, ATom_data)
model = fns.train(in_train, out_train)
out_pred, mse, r2 = fns.test(model, in_test, out_test)
fns.show(out_test, out_pred, mse, r2)
