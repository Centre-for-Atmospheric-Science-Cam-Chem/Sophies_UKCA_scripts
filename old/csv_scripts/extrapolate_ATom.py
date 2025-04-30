'''
Name: Sophie Turner.
Date: 6/9/2024.
Contact: st838@cam.ac.uk
Try to predict J rates at unseen UKCA gridboxes using a model trained on ATom J rates.
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
UKCA_train_file = dir_path + 'nudged_J_outputs_for_ATom/UKCA_hourly_all.csv'
UKCA_rand_file = dir_path + 'nudged_J_outputs_for_ATom/random_points.csv'

ATom_data = pd.read_csv(ATom_file)
UKCA_train_data = pd.read_csv(UKCA_train_file) 
UKCA_rand_data = pd.read_csv(UKCA_rand_file)

for dataset in [ATom_data, UKCA_train_data, UKCA_rand_data]:
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
	 
input_name = phys_correct

for target_name in J_all:
  target_name = [target_name]
  print('\nTarget:', target_name)

  # Which datasets to take items from.
  input_data = UKCA_train_data
  target_data = ATom_data

  # Make and train the model using ATom targets.
  input_name, target_name, input_data, target_data = fns.prep_data(input_name, target_name, input_data, target_data, J_all)
  in_train, in_test, out_train, out_test = fns.set_params(target_name, input_name, target_data, input_data)
  model = fns.train(in_train, out_train)
  
  # Apply the model to other UKCA input data.
  input_data = UKCA_rand_data
  target_data = UKCA_rand_data
  inputs = input_data[input_name] 
  targets = target_data[target_name]
  out_pred = model.predict(inputs)
  
  # Make a plot.
  plt.scatter(targets, out_pred)
  
  # Force axes to be identical.
  fns.force_axes()
  
  plt.title(f'{target_name[0].split()[0]} at random UM grid points')
  plt.xlabel('UKCA J rate / s\u207b\u00b9')
  plt.ylabel('Linear regression of UKCA & ATom J rate / s\u207b\u00b9')
  plt.show() 
  plt.close()  
  
  # Show another kind of plot.
  plt.scatter(UKCA_rand_data['SOLAR ZENITH ANGLE'], targets, label=f'UKCA J rate', alpha=0.5)
  plt.scatter(UKCA_rand_data['SOLAR ZENITH ANGLE'], out_pred, label=f'J rate from linear regression of ATom known J rates', alpha=0.5)
  plt.legend()
  plt.title(f'Predictions of UKCA {target_name[0].split()[0]}')
  plt.xlabel('Solar zenith angle / deg')
  plt.ylabel('J rate / s\u207b\u00b9')
  plt.show() 
  plt.close()

