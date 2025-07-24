import pandas as pd
import glob

# File paths.
UKCA_files = glob.glob('/scratch/st838/netscratch/nudged_J_outputs_for_ATom/UKCA_hourly*.csv')
ATom_files = glob.glob('/scratch/st838/netscratch/ATom_MER10_Dataset/ATom_hourly*.csv')

for files in [UKCA_files, ATom_files]:
  for each_file in files:
    data = pd.read_csv(each_file, index_col=0)
    data = data.rename(columns={'CLOUD %':'CLOUD FRACTION'})    
    data.to_csv(each_file)

