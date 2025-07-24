'''
Name: Sophie Turner.
Date: 12/1/2024.
Contact: st838@cam.ac.uk
Script to investigate and analyse apparent errors in UM solar zenith angle.
Used on Cambridge chemistry department's atmospheric servers. 
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize 
from sklearn.metrics import r2_score
import comparison_fns_shared as fns


def make_filename_2_fields(name):
  name = name.replace(',', '')
  name = name.replace('ATom - UKCA', 'DIFF')
  name = name.replace(' ', '_')
  return(name)


def data_diff(dataATom, dataUKCA):
  # Get the difference of UKCA from ATom for a field to check against other vars.
  diff = dataATom - dataUKCA
  name = f'ATom - UKCA {dataATom.name}'
  return diff, name


def plot_timeseries_2_fields(path, field1, field2, field2ATom=None):
  # Works best with data from one flight and 2 fields.
  date, times = fns.split_date_time(field1)
  if field2ATom is not None:
    # Instead of directly comparing 2 fields, compare 1 field against the diff in another field.
    field2, field2name = data_diff(field2ATom, field2) 
  else:
    field2name = field2.name
  title = f'{field1.name}, {field2name}'  
  name = make_filename_2_fields(title)
  plt.figure()
  plt.title(f'UKCA {title} {date}')
  x = [t[0:5] for t in times] # Cut the extra 00s for seconds off the time labels.
  y1 = field1
  y2 = field2
  plt.plot(x, y1, label=field1.name)
  plt.plot(x, y2, label=field2name)
  plt.xlabel('time')
  plt.legend()
  plt.savefig(f'{path}/{name}_{date}_ts.png')
  #plt.show()    
  plt.close()
  
  
def plot_corr_2_fields(path, field1, field2, field2ATom=None):
  # Works with data for 2 fields.
  if field2ATom is not None:
    # Instead of directly comparing 2 fields, compare 1 field against the diff in another field.
    field2, field2name = data_diff(field2ATom, field2) 
  else:
    field2name = field2.name
  title = f'{field1.name}, {field2name}' 
  name = make_filename_2_fields(title)
  plt.figure(figsize=(7,5))
  r2 = round(r2_score(field2, field1), 2)
  plt.figtext(0.25, 0.75, f'r\u00b2 = {r2}')
  x = field2
  y = field1
  plt.scatter(x, y)
  # Line of best fit.
  plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), c='black', linestyle='dotted')
  #line_eqn = get_line_eqn(x, y)
  # Display it on chart with R2 score.
  #plt.text(-5, 60, f'r\u00b2 = {r2}\n{line_eqn}')
  plt.xlabel(field2name) 
  plt.ylabel(field1.name)
  # The title depends on whether we're looking at all flights or 1 flight.
  if len(field1) < 24:
    date, _ = fns.split_date_time(field1)
    plt.title(f'UKCA {title} {date}') 
    path = f'{path}/{name}_{date}_corr.png'
  else:
    plt.title(f'UKCA {title}') 
    path = f'{path}/{name}_corr.png'
  plt.savefig(path)
  #plt.show() 
  plt.close()


# File paths.
path = '/scratch/st838/netscratch/'
out_dir = path + 'analysis/sza'
ATom_dir = path + 'ATom_MER10_Dataset'
UKCA_dir = path + 'nudged_J_outputs_for_ATom'
ATom_file = f'{ATom_dir}/ATom_hourly_all.csv'
UKCA_file = f'{UKCA_dir}/UKCA_hourly_all.csv'
ATom_daily_files = glob.glob(ATom_dir + '/ATom_hourly_20*.csv') 
 
ATom_all = pd.read_csv(ATom_file, index_col=0)
UKCA_all = pd.read_csv(UKCA_file, index_col=0) 

# Brief check indicated there might be a relationship between SZA and longitude.
# Note that longitude is related to Earth/solar time so could actually be time.

sza = 'SOLAR ZENITH ANGLE'
lon = 'LONGITUDE' 
ATom_field = ATom_all[sza]
UKCA_field = UKCA_all[sza] 

# See if there is a detectable correlation.
diff_sza, _ = data_diff(ATom_field, UKCA_field)
# Normalise.
diff_sza_norm = normalize([diff_sza])
lons_norm = normalize([UKCA_all[lon]]) 
r2 = round(r2_score(diff_sza_norm[0], lons_norm[0]), 2)
print(f'r\u00b2 of normalised SZA diffs and longitude = {r2}')
r2 = round(r2_score(lons_norm[0], diff_sza_norm[0]), 2)
print(f'r\u00b2 of normalised longitude and SZA diffs = {r2}')

# See if there is a correlation to abs error.
diff_sza = abs(diff_sza)
# Normalise.
diff_sza_norm = normalize([diff_sza])
r2 = round(r2_score(diff_sza_norm[0], lons_norm[0]), 2)
print(f'r\u00b2 of normalised absolute SZA diffs and longitude = {r2}')

# Plot all of them to visualise trends.
# Look at each flight.
# Plot UKCA and ATom.
for ATom_day_file in ATom_daily_files:
  ATom_day = pd.read_csv(ATom_day_file, index_col=0)
  if len(ATom_day) >= 2:
    date, _ = fns.split_date_time(ATom_day)
    UKCA_day_file = f'{UKCA_dir}/UKCA_hourly_{date}.csv'
    UKCA_day = pd.read_csv(UKCA_day_file, index_col=0)
    fns.plot_location(ATom_day, UKCA_day, out_dir)
    # Plot the field.
    fns.plot_timeseries(ATom_day[sza], UKCA_day[sza], out_dir)
    fns.plot_corr(out_dir, ATom_day[sza], UKCA_day[sza], UKCA_day['LATITUDE'], remove_null=True) 
    # Plot both fields together for UKCA.
    plot_timeseries_2_fields(out_dir, UKCA_day[lon], UKCA_day[sza])
    plot_corr_2_fields(out_dir, UKCA_day[lon], UKCA_day[sza])
    # Plot diff between datasets.
    plot_timeseries_2_fields(out_dir, UKCA_day[lon], UKCA_day[sza], ATom_day[sza])
    plot_corr_2_fields(out_dir, UKCA_day[lon], UKCA_day[sza], ATom_day[sza])      
