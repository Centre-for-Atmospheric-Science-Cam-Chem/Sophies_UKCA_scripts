'''
Name: Sophie Turner.
Date: 30/10/2023.
Contact: st838@cam.ac.uk
Script to compare pre-procecced ATom and UKCA data and see differences.
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
from numpy import ones,vstack
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import pylab
import cartopy.crs as ccrs
from sklearn.metrics import r2_score


def view_values(data, name):
  # data: list or array of data points.
  num_values = len(data)
  values, counts = np.unique(data, return_counts=True)
  if any(val > 1 for val in counts):
    print(f'\n{name} contains mostly:')  
    print('value, count, % of data:')
    counts_temp = counts.copy()
    for _ in range(10):
      i = np.argmax(counts_temp)
      if counts_temp[i] > 1:
        print(values[i], counts_temp[i], (counts_temp[i]/num_values)*100)
        counts_temp[i] = 0
      else:
        print('All the other values occur only once.')
        break
      

def view_basics(data, name):
  # data: list or np array.
  largest = np.max(data)
  smallest = np.min(data)
  avg = np.mean(data)
  print(f'\nThe {name} range from {smallest} to {largest}.')
  print(f'Mean value of {name}:', avg)
  return(largest, smallest, avg)


def simple_diff(value_1, value_2, data_1_name, data_2_name, values_name, field_name):
  # Compare 2 numbers.
  # value_1 and value_2 are the nums to compare.
  # data_1_name and data_2_name are the names of the datasets.
  # values_name is what the values are e.g. 'largest', 'smallest', 'average'.
  # field_name describes what the stash item is.
  if value_1 != value_2:
    diff = value_2/value_1  
    print(f'The {values_name} {field_name} value from {data_2_name} is {round(diff, 2)} x the {values_name} value from {data_1_name}.')
  else:
    print(f'The {values_name} {field_name} values from both datasets are the same.')


def diffs(data_1, data_2, name_1, name_2):
  # data1 - data2 to see differences.
  num_values = len(data_1)
  diff = data_2 - data_1
  rel_diff = diff/data_1 * 100
  # how many data are different?
  num_diff = np.count_nonzero(diff)
  print(f'\n{num_diff} of {name_2} values are different to the corresponding {name_1} values.')
  # what % of the data are different?
  print(f'{(num_diff/num_values)*100}% of the data are different.')
  # by how much do they differ?
  view_basics(diff, 'differences')
  view_basics(rel_diff, 'relative differences %')
  
  
def split_date_time(data):
  date_time = data.index[0]
  date = date_time.split()[0]
  times = []
  for item in data.index:
    date_time = item.split()
    times.append(date_time[1])
  return(date, times)
  
  
def make_title(name):
  if name != 'RELATIVE HUMIDITY' and name != 'SOLAR ZENITH ANGLE':
    name = name.split()[0]
  return(name)
  
  
def get_line_eqn(x, y):
  # Calculate eqn of a straight line.
  # Slope.
  x1, y1 = x[0], y[0]
  x2, y2 = x[4], y[4]
  m = (y2-y1)/(x2-x1)
  # Y intercept.
  # c = y - (m * x).

  # Formulate y=mx+c.
  sign = ''
  if c >= 0:
    sign='+ '
  eqn = f'y = {m} x {sign}{c}'
  return(eqn)  
  
  
def plot_timeseries(dataATom, dataUKCA):
  # Works best with data from one flight, and one field.
  date, times = split_date_time(dataATom)
  title = make_title(dataATom.name)
  plt.figure()
  plt.title(f'{title} ALONG FLIGHT PROGRESSION, {date}')
  x = times
  y1 = dataATom
  y2 = dataUKCA
  plt.plot(x, y1, label='ATom')
  plt.plot(x, y2, label='UKCA')
  plt.xlabel('TIME')
  plt.ylabel(dataATom.name)
  plt.legend()
  plt.show()    
  
  
def plot_corr(dataATom, dataUKCA, lat):
  # Works with data for one field.
  # Make axes the same scale.
  title = make_title(dataATom.name)
  plt.figure()
  r2 = round(r2_score(dataUKCA, dataATom), 1)
  plt.figtext(0.25, 0.75, f'r\u00b2 = {r2}')
  plt.title(f'CORRELATION OF {title} FROM ATOM AND UKCA') 
  x = dataUKCA
  y = dataATom
  plt.scatter(x, y, c=lat)
  # Line of best fit.
  plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), c='black', linestyle='dotted')
  #line_eqn = get_line_eqn(x, y)
  # Display it on chart with R2 score.
  #plt.text(-5, 60, f'r\u00b2 = {r2}\n{line_eqn}')
  plt.xlabel(f'UKCA {title}') 
  plt.ylabel(f'ATom {title}')
  plt.colorbar(label='Latitude / degrees North')
  plt.show() 
  

def plot_diff(dataATom, dataUKCA, path):
  # Works best with data from all times and all flights at once, and one field.
  title = make_title(dataATom.name)
  plt.figure()
  plt.title(f'UKCA DIFFERENCE TO ATOM FOR {title}')
  diff = dataUKCA - dataATom
  # Prevent divide by zero by treating zero as the smallest number > 0.
  smallest_ATom = np.min(dataATom[np.nonzero(dataATom)[0]])
  smallest_UKCA = np.min(dataUKCA[np.nonzero(dataUKCA)[0]])
  smallest = smallest_UKCA
  if smallest_UKCA > smallest_ATom:
    smallest = smallest_ATom
  dataATom[dataATom == 0.0] = smallest
  dataUKCA[dataUKCA == 0.0] = smallest 
  rel_diff = diff/dataATom * 100
  plt.hist(rel_diff)
  plt.xlabel('% difference')
  plt.ylabel('Number of data points')
  #plt.savefig(f'{path}/diff_{title}.png')
  plt.show()
  
  
def plot_location(data1, data2):
  date, _ = split_date_time(data1)
  fig = pylab.figure(dpi=150)
  ax = pylab.axes(projection=ccrs.PlateCarree(central_longitude=310.0))
  ax.stock_img()
  pylab.title(f'Locations of selected flight path points, {date}')
  alt = data1['ALTITUDE m']
  x = data1['LONGITUDE']
  y = data1['LATITUDE']
  pylab.scatter(x+50, y, s=20, c=alt, cmap='Reds', marker='_', label='ATom')
  alt = data2['ALTITUDE m']
  x = data2['LONGITUDE']
  y = data2['LATITUDE']
  pylab.scatter(x+50, y, s=20, marker='|', c=alt, cmap='Reds', label='UKCA')
  pylab.legend()
  pylab.colorbar(label='Altitude / m')
  pylab.show()
  
 
# File paths.
path = '/scratch/st838/netscratch/'
out_dir = path + 'analysis'
ATom_dir = path + 'ATom_MER10_Dataset'
UKCA_dir = path + 'nudged_J_outputs_for_ATom'
ATom_file = f'{ATom_dir}/ATom_hourly_all.csv'
UKCA_file = f'{UKCA_dir}/UKCA_hourly_all.csv'
ATom_daily_files = glob.glob(ATom_dir + '/ATom_hourly_20*.csv') 
UKCA_daily_files = glob.glob(UKCA_dir + '/UKCA_hourly_20*.csv') 
 
ATom_all = pd.read_csv(ATom_file, index_col=0)
UKCA_all = pd.read_csv(UKCA_file, index_col=0) 

# test
field = 'JO3 O2 O1D'
plot_diff(ATom_all[field], UKCA_all[field], out_dir)
plot_corr(ATom_all[field], UKCA_all[field], UKCA_all['LATITUDE'])
 
#for field in ATom_all.columns:
#  plot_diff(ATom_all[field], UKCA_all[field], out_dir)   

#plot_location(ATom_data, UKCA_data)
#plot_timeseries(ATom_data[field], UKCA_data[field])

