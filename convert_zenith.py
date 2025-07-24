# Convert radians to degrees if it was not done already.

import pandas as pd
import glob
from math import pi

# File paths.
path = '/scratch/st838/netscratch/nudged_J_outputs_for_ATom/'
UKCA_files = glob.glob(path + '/UKCA_hourly*.csv') 

for UKCA_file in UKCA_files:
  data = pd.read_csv(UKCA_file, index_col=0)
  zen = data['SOLAR ZENITH ANGLE']
  # Conversion.
  zen = zen * 180 / pi
  data['SOLAR ZENITH ANGLE'] = zen  
  data.to_csv(UKCA_file)
