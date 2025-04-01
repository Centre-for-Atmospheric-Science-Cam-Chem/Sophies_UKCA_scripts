# Test to see if there is a problem with the SZA given in the ATom data 
# due to UTC conversion issues.
# Uses Pysolar to calculate SZA independently.

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pysolar.solar import *
from glob import glob
from sklearn.preprocessing import Normalizer


def conversions(data):
  # Convert pressure from hectopascals to Pascals.
  data['PRESSURE hPa'] = data['PRESSURE hPa'] * 100
  data = data.rename(columns={'PRESSURE hPa':'PRESSURE Pa'})
  # Distinguish between reported and independently calculated SZAs.
  data = data.rename(columns={'SOLAR ZENITH ANGLE':'SZA REPORTED'})
  return(data)
  
  
def calc_SZA(data):
  # Loop through the timesteps and calculate SZAs from physical data.
  # data is for one flight.
  SZAs, times = [], []
  for hour in data.index:
    # For each hourly timestep.    
    # Format time nicely for plots.
    t = hour.time()
    t = t.isoformat(timespec='minutes')
    times.append(t)
    # Get the parameters for the pysolar calculation.
    dt = hour.replace(tzinfo=datetime.timezone.utc)
    lat = data.loc[hour]['LATITUDE']
    lon = data.loc[hour]['LONGITUDE']
    alt = data.loc[hour]['ALTITUDE m']
    temp = data.loc[hour]['TEMPERATURE K']
    pres = data.loc[hour]['PRESSURE Pa']
    # Get the angle of the sun from the horizon.
    a = get_altitude(lat, lon, dt, alt, temp, pres)
    SZAs.append(90 - a) # Conversion to SZA assumes that we are at sea level.
  # Append the calculated SZAs to the datasets as columns for easy indexing later.
  data.insert(15, 'SZA CALCULATED', SZAs)
  return(data, times)


ATom_files = glob('/scratch/st838/netscratch/ATom_MER10_Dataset/ATom_hourly_20*.csv')
out_path = '/scratch/st838/netscratch/analysis/SZA_investigations'

# Look at each flight.
for ATom_file in ATom_files:
  # Read time in datetime format.
  ATom_data = pd.read_csv(ATom_file, parse_dates=['TIME'], index_col=0) 
  # There are a lot of flights. Just look at the longest ones.
  if len(ATom_data) >= 8:
    # Find the matching data.
    name = ATom_file[-14:]
    UKCA_file = f'/scratch/st838/netscratch/nudged_J_outputs_for_ATom/UKCA_hourly_{name}'
    UKCA_data = pd.read_csv(UKCA_file, parse_dates=['TIME'], index_col=0) 

    # Conversions to units needed for Pysolar (pressure in Pascals) and rename.
    ATom_data = conversions(ATom_data)
    UKCA_data = conversions(UKCA_data) 
    
    # Undo the conversion from radians to degrees on UM SZAs.
    rads = np.cos(UKCA_data['SZA REPORTED'] * (pi / 180.0))
    UKCA_data.insert(15, 'SZA NOT CONVERTED', rads)
  
    # Calculate SZAs from physical data.
    ATom_data, _ = calc_SZA(ATom_data)
    UKCA_data, times = calc_SZA(UKCA_data)
    
    # Get date from datetimes.
    date = UKCA_data.index[0].date()
  
    plt.figure(figsize=(8,5))
    plt.title(f'Solar zenith angles along flight path on {date}')
    plt.plot(times, ATom_data['SZA REPORTED'], label='ATom reported SZA')
    plt.plot(times, ATom_data['SZA CALCULATED'], label='ATom independently calculated SZA (at surface)')
    plt.plot(times, UKCA_data['SZA REPORTED'], label='UM output SZA (at surface)')
    plt.plot(times, UKCA_data['SZA CALCULATED'], label='UM independently calculated SZA (at surface)')
    #plt.plot(times, UKCA_data['SZA NOT CONVERTED'], label='UM SZA without conversion')
    plt.xlabel('UTC time')
    plt.ylabel('Solar zenith angle / degrees')
    plt.legend()
    plt.savefig(f'{out_path}/SZAs_{date}_ts.png')
    plt.show()    
    plt.close()  
    
    # Look at how the difference in SZA corresponds to the difference in J rates.    
    diff_SZA = Normalizer().fit_transform([abs(UKCA_data['SZA REPORTED'] - ATom_data['SZA REPORTED'])]) 
    J_all = ATom_data.loc[:,ATom_data.columns.str.startswith('J')].columns.tolist()
    
    # Plot it.
    plt.figure(figsize=(22,7.5))
    plt.title(f'Differences in solar zenith angle and photolysis rates along flight path on {date}')
    diff_J_all = []
    for J in J_all:
      diff_J = Normalizer().fit_transform([abs(UKCA_data[J] - ATom_data[J])])
      diff_J_all.append(np.squeeze(diff_J))
      label=f'Difference between UKCA and ATom {J.split()[0]}'
      plt.plot(times, np.squeeze(diff_J), linestyle='dotted', linewidth=1, label=label)
    plt.plot(times, np.mean(diff_J_all, axis=0), linestyle='dashed', linewidth=2, label='Average difference between UKCA and ATom J rates')
    plt.plot(times, np.squeeze(diff_SZA), linewidth=2, label='Difference between UM and ATom SZA')
    plt.xlabel('UTC time')
    plt.ylabel('Normalised absolute difference')
    plt.legend()
    plt.savefig(f'{out_path}/diff_all_{date}_ts.png')
    plt.show()    
    plt.close() 
  
