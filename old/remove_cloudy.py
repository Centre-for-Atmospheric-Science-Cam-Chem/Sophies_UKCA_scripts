#!/usr/bin/env python

import pandas as pd
import numpy as np

def remove_cloudy(data1, data2):
  # Remove entries where the amount of cloud in UKCA and ATom differs by more than 10%.
  drops = []
  for timestep in data1.index:
    cloud1 = data1.loc[timestep]['CLOUD %']
    cloud2 = data2.loc[timestep]['CLOUD %']
    if abs(cloud1 - cloud2) > 10:
      drops.append(timestep)
  data1 = data1.drop(index=drops)
  data2 = data2.drop(index=drops)
  return(data1, data2)
  
    
ATom_file = '/scratch/st838/netscratch/ATom_MER10_Dataset/ATom_hourly_2016-08-20.csv'
ATom_data = pd.read_csv(ATom_file, index_col=0)

UKCA_file = '/scratch/st838/netscratch/nudged_J_outputs_for_ATom/UKCA_hourly_2016-08-20.csv'
UKCA_data = pd.read_csv(UKCA_file, index_col=0)

remove_cloudy(ATom_data, UKCA_data)
