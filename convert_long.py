import pandas as pd
import glob

# File paths.
path = '/scratch/st838/netscratch/nudged_J_outputs_for_ATom/'
UKCA_files = glob.glob(path + '/UKCA_hourly*.csv') 

for UKCA_file in UKCA_files:
  data = pd.read_csv(UKCA_file, index_col=0)
  lon = data['LONGITUDE']

  # Conversion.
  lon[lon > 180] = lon - 360
  data['LONGITUDE'] = lon

  data.to_csv(UKCA_file)

