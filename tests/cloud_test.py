import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Datasets we want to look at.
ATom_file_full = '/scratch/st838/netscratch/ATom_MER10_Dataset/photolysis_data.csv'
ATom_file_matched = '/scratch/st838/netscratch/ATom_MER10_Dataset/ATom_hourly_all.csv' 
UKCA_file_matched = '/scratch/st838/netscratch/nudged_J_outputs_for_ATom/UKCA_hourly_all.csv'

# Fields we want to look at.
flag = 'CloudFlag_AMS'
indicator = 'cloudindicator_CAPS' 
cloud = 'CLOUD %'
J = 'jCH2O_H_HCO_CAFS'

# Pick what we want to see.
field1 = flag
file1 = ATom_file_full
field2 = J
file2 = ATom_file_full

# Read the dataset.
data1 = pd.read_csv(file1)
'''
# Look at the values in a field.
values = pd.unique(data1[field1])
print(np.sort(values))
plt.hist(data1[field1])
plt.show()
'''
# compare two fields.
data2 = pd.read_csv(file2)

plt.scatter(data1[field1], data2[field2])
plt.xlabel(field1)
plt.ylabel(field2)
plt.show()







