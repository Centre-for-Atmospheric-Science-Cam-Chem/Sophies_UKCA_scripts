'''
Name: Sophie Turner.
Date: 19/1/2023.
Contact: st838@cam.ac.uk
Try to predict ATom J rates using UKCA data as inputs.
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
ATom_file = dir_path + 'ATom_MER10_Dataset/ATom_hourly_all.csv'
UKCA_file = dir_path + 'nudged_J_outputs_for_ATom/UKCA_hourly_all.csv'

ATom_data = pd.read_csv(ATom_file)
UKCA_data = pd.read_csv(UKCA_file) 

for dataset in [ATom_data, UKCA_data]:
  # Get time info as float.
  fns.time_to_s(dataset)
  
# Set up data structures for different experiments.
phys_all = ['SECONDS', 'TEMPERATURE K', 'LATITUDE', 'LONGITUDE', 'ALTITUDE m', 'CLOUD %', 
            'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY', 'PRESSURE hPa']
J_all = ['JO3 O2 O1D', 'JNO2 NO O3P', 'JH2O2 OH OH', 'JNO3 NO O2', 'JCH2O H HCO', 
         'JCH2O H2 CO', 'JPROPANAL CH2CH3 HCO', 'JMEONO2 CH3O NO2', 'JHOBR OH BR']
	 
input_name = ['JNO2 NO O3P']
target_name = ['JNO2 NO O3P']

ATom_data, UKCA_data = fns.remove_tiny(target_name, ATom_data, UKCA_data)
UKCA_data, ATom_data = fns.remove_tiny(input_name, UKCA_data, ATom_data)

input_data = UKCA_data
target_data = ATom_data

in_train, in_test, out_train, out_test = fns.set_params(target_name, input_name, target_data, input_data)
model = fns.train(in_train, out_train)
out_pred, mse, r2 = fns.test(model, in_test, out_test)
fns.show(out_test, out_pred, mse, r2)

