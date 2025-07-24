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

import pandas as pd
import matplotlib.pyplot as plt
import prediction_fns_pandas as fns

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
phys_all = ['SECONDS', 'TEMPERATURE K', 'LATITUDE', 'LONGITUDE', 'ALTITUDE m', 'CLOUD FRACTION', 
            'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY', 'PRESSURE hPa']
phys_reduced = ['TEMPERATURE K', 'PRESSURE hPa', 'CLOUD FRACTION', 'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY']
phys_min = ['PRESSURE hPa', 'CLOUD FRACTION', 'SOLAR ZENITH ANGLE']
# Only for use when using UKCA and ATom together. Remove fields that don't compare well.
phys_correct = ['SECONDS', 'TEMPERATURE K', 'LATITUDE', 'LONGITUDE', 'ALTITUDE m', 
                'SOLAR ZENITH ANGLE', 'PRESSURE hPa'] 
# Use this J_all if using UKCA and ATom together.	    
J_all = ['JO3 O2 O1D', 'JNO2 NO O3P', 'JH2O2 OH OH', 'JNO3 NO O2', 'JCH2O H HCO', 
         'JCH2O H2 CO', 'JPROPANAL CH2CH3 HCO', 'JMEONO2 CH3O NO2', 'JHOBR OH BR']	
# Use this J_all if only using one dataset.
#J_all = UKCA_data.loc[:,UKCA_data.columns.str.startswith('J')].columns.tolist()
	 
input_name = phys_correct

for target_name in J_all:
  target_name = [target_name]
  print('\nTarget:', target_name)

  # Which dataset to take items from.
  input_data = ATom_data.copy()
  target_data = ATom_data.copy()

  #input_data, target_data = fns.remove_tiny(input_name, input_data, target_data)
  input_name, target_name, input_data, target_data = fns.prep_data(input_name, target_name, input_data, target_data, J_all)
  in_train, in_test, out_train, out_test_ATom = fns.set_params(target_name, input_name, target_data, input_data)
  model = fns.train(in_train, out_train)
  out_pred_ATom, mse, mape, r2_ATom = fns.test(model, in_test, out_test_ATom)
  
  # Which dataset to take items from.
  input_data = UKCA_data.copy()
  target_data = ATom_data.copy()

  #input_data, target_data = fns.remove_tiny(input_name, input_data, target_data)
  input_name, target_name, input_data, target_data = fns.prep_data(input_name, target_name, input_data, target_data, J_all)
  in_train, in_test, out_train, out_test_UKCA = fns.set_params(target_name, input_name, target_data, input_data)
  model = fns.train(in_train, out_train)
  out_pred_UKCA, mse, mape, r2_UKCA = fns.test(model, in_test, out_test_UKCA)
  #fns.show(out_test, out_pred, mse, r2)

  plt.scatter(out_test_ATom, out_pred_ATom, label=f'Model trained on ATom physical inputs, R\u00b2={round(r2_ATom, 2)}', alpha=0.5) # alpha=0.01 for large data.
  plt.scatter(out_test_UKCA, out_pred_UKCA, label=f'Model trained on UKCA physical inputs, R\u00b2={round(r2_UKCA, 2)}', alpha=0.5)
  
  # Force axes to be identical.
  plt.axis('square')
  xticks, xlabels = plt.xticks()
  yticks, ylabels = plt.yticks()
  plt.axis('auto')
  if len(yticks) > len(xticks):
    tix = yticks
    lab = ylabels
  else:
    tix = xticks
    lab = xlabels
  plt.xticks(ticks=tix, labels=lab)
  plt.yticks(ticks=tix, labels=lab) 
  
  plt.legend()
  plt.title(f'Predictions of ATom {target_name[0].split()[0]}')
  plt.xlabel('ATom J rate / s\u207b\u00b9')
  plt.ylabel('Linear regression J rate / s\u207b\u00b9')
  plt.show() 
  plt.close()
