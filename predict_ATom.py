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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime


def standard_names(data):
  # Standardise the names of ATom fields into a simple format.
  for name_old in data.iloc[:,4:-1].columns:
    name_new = name_old.replace('CAFS', '')
    name_new = name_new.replace('_', ' ')
    name_new = name_new.upper()  
    name_new = name_new.strip()
    data = data.rename(columns={name_old:name_new})
  return(data)
  

# File paths.
dir_path = '/scratch/st838/netscratch/'
ATom_dir = dir_path + 'ATom_MER10_Dataset/'
ATom_file = ATom_dir + 'photolysis_data.csv'

# Open ATom dataset, already partially pre-processed.
ATom_data = pd.read_csv(ATom_file)
ATom_data = ATom_data.rename(columns={'UTC_Start_dt':'TIME', 'T':'TEMPERATURE', 'G_LAT':'LATITUDE', 
                                      'G_LONG':'LONGITUDE', 'G_ALT':'ALTITUDE', 'Pres':'PRESSURE',
				      'cloudindicator_CAPS':'CLOUD %'})    
ATom_data = standard_names(ATom_data)  

# Convert time info into datetime and float types.
# Formatted as str like '2017-10-04 22:55:00'.
seconds = []
for row in ATom_data.index: 
  time_str = ATom_data.loc[row]['TIME']
  time_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
  second = time_dt.timestamp()
  seconds.append(second)
ATom_data.insert(1, 'SECONDS', seconds)

# Set up data structures for different experiments.
not_used = ['CLOUDFLAG AMS', 'JO3 DNWFRAC', 'JNO2 DNWFRAC']
ATom_data = ATom_data.drop(columns=not_used)
J_all = ATom_data.loc[:,ATom_data.columns.str.startswith('J')]
phys_all = ['SECONDS', 'TEMPERATURE', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 
            'PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY']			  
phys_reduced = ['TEMPERATURE', 'PRESSURE', 'CLOUD %', 'SOLAR ZENITH ANGLE', 'RELATIVE HUMIDITY']

# Choose parameters for training. 1 J from all physics.
target_name = 'JNO2 NO O3P'
#target_name = 'JH2O2 OH OH'
input_name = phys_all
all_used = input_name.copy().append(target_name)

# Remove relevant rows with empty fields.
# Removes up to 92% of the data!
ATom_data = ATom_data.dropna(axis=0, subset=all_used) 

target = ATom_data[target_name]
inputs = ATom_data[input_name]

# Split data (almost randomly).
in_train, in_test, out_train, out_test = train_test_split(inputs, target, test_size=0.2, random_state=6)

# Set up simple model.
mlr = linear_model.LinearRegression()
mlr.fit(in_train, out_train)

# Try it out.
out_pred = mlr.predict(in_test)

# See how well it did.
mse = mean_squared_error(out_test, out_pred)
r2 = r2_score(out_test, out_pred)
print(f'Mean squared error for {target_name} predictions = {round(mse, 1)}')
print(f'Coefficient of determination for {target_name} predictions = {round(r2, 1)}')
plt.scatter(out_test, out_pred)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()
