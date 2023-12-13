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


def remove(dataATom, dataUKCA, other=None, null=True, zero=False):
  # other: additional data to remove from.
  # null, zero: remove nulls and/or zeros.
  remove = []
  # Only the ATom data can contain nulls. 
  if null:
    remove.append(dataATom[dataATom.isna()].index)
  if zero:
    remove.append(dataATom[dataATom == 0.0].index)  
    remove.append(dataUKCA[dataUKCA == 0.0].index)  
  # Remove those from both datasets and other fields if needed.
  for indices in remove:
    for idx in indices:
      if idx in dataATom.index:
        dataATom = dataATom.drop(index=idx)
        dataUKCA = dataUKCA.drop(index=idx)
        if other is not None:
          other = other.drop(index=idx)
  return(dataATom, dataUKCA, other)


def rel_diff_no_zero(data1, data2):
  # Relative difference. 
  # Remove zeros from divisor and corresponding dividend elements to prevent arithmetic errors.
  data1 = data1[np.nonzero(data1)[0]]
  data2 = data2[np.nonzero(data1)[0]]
  diff = data2 - data1
  rel_diff = diff/data1 * 100
  return(rel_diff)


def scale_zero(data1, data2):
  # Prevent divide by zero by treating zero as the smallest number that > 0.
  # Can cause scientific error. Only use if you can't remove nonzero values.
  smallest1 = np.min(data1[np.nonzero(data1)[0]])
  smallest2 = np.min(data2[np.nonzero(data2)[0]])
  smallest = smallest2
  if smallest2 > smallest1:
    smallest = smallest1
  data1[data1 == 0.0] = smallest
  data2[data2 == 0.0] = smallest
  return(data1, data2) 
  
  
def too_diff(data1, data2):
  # Determine if the datasets are too different to be able to view in the same scale.
  means = [np.mean(data1), np.mean(data2)]
  diff = np.max(means) / np.min(means)
  # If one mean is 5x higher than the other.
  if diff > 5:
    too = True
  else:
    too = False
  return(too)  
  

def view_values(data, name, round_by=10):
  # data: list or array of data points.
  # name ~ 'ATom temperature K'
  num_values = len(data)
  values, counts = np.unique(data, return_counts=True)
  if any(val > 1 for val in counts):
    out_str = f'\n{name} contains mostly:'  
    out_str += '\nvalue, count, % of data:'
    counts_temp = counts.copy()
    for _ in range(10):
      i = np.argmax(counts_temp)
      if counts_temp[i] > 1:
        out_str += f'\n{round(values[i], 10), counts_temp[i], round((counts_temp[i]/num_values)*100, 1)}'
        counts_temp[i] = 0
      else:
        out_str += '\nAll the other values occur only once.'
        break
    return(out_str)
  elif round_by > 0:
    round_by -= 1
    return(view_values(round(data, round_by), name, round_by))
        

def view_basics(data, name):
  # data: list or np array.
  largest = np.max(data)
  smallest = np.min(data)
  avg = np.mean(data)
  str_out = f'\nThe {name} range from {round(smallest, 10)} to {round(largest, 10)}.'
  str_out += f'\nMean value of {name} = {round(avg, 10)}.'
  return(str_out, largest, smallest, avg)


def simple_diff(value1, value2, data1_name, data2_name, values_name, field_name):
  # Compare 2 numbers.
  # value1 and value2 are the nums to compare.
  # data1_name and data2_name are the names of the datasets.
  # values_name is what the values are e.g. 'largest', 'smallest', 'average'.
  # field_name describes what the stash item is.
  if value1 != value2:
    diff = value2/value1  
    str_out = f'\nThe {values_name} {field_name} value from {data2_name} is {round(diff, 10)} x the {values_name} value from {data1_name}.'
  else:
    str_out = f'\nThe {values_name} {field_name} values from both datasets are the same.'
  return(str_out)


def make_title(name):
  if name != 'RELATIVE HUMIDITY' and name != 'SOLAR ZENITH ANGLE':
    name = name.split()[0]
  return(name)


def make_sentence(name):
  if name[0] != 'J':
    name = name.lower()
  name = name.replace(' k', ' K')
  name = name.replace('hpa', 'hPa')
  return(name)
  
  
def make_filename(name):
  name = make_title(name)
  name = make_sentence(name)
  name = name.replace(' ', '_')
  return(name)


def diffs(data1, data2, data1_name, data2_name, path):
  # data1_name ~ 'ATom'
  # data1 - data2 to see differences.
  num_values = len(data1)
  diff = round(data2 - data1, 10)
  # how many data are different?
  num_diff = np.count_nonzero(diff)
  # Relative difference.
  rel_diff = rel_diff_no_zero(data1, data2)
  # Put the answers in a file.
  name = make_filename(data1.name)
  printout = open(f'{path}/{name}_diff.txt', 'w')
  name = make_sentence(data1.name)  
  diff_str, _, _, avg1 = view_basics(data1, f'ATom {name}')
  printstr = f'\n{diff_str}'
  diff_str, _, _, avg2 = view_basics(data2, f'UKCA {name}')
  printstr += f'\n{diff_str}' 
  avg_str = simple_diff(avg1, avg2, data1_name, data2_name, 'mean', name)
  printstr += f'\n{avg_str}'
  vals_str = view_values(data1, f'ATom {name}')
  printstr += f'\n{vals_str}'
  vals_str = view_values(data2, f'UKCA {name}')
  printstr += f'\n{vals_str}'
  printstr += f'\n{num_diff} of {data2_name} {name} values are different to the corresponding {data1_name} {name} values.'
  # what % of the data are different?
  printstr += f'\n{round((num_diff/num_values)*100, 2)}% of the data are different.'
  # by how much do they differ?
  diff_str = view_basics(diff, 'differences')
  rel_str = view_basics(rel_diff, 'relative differences %')
  printstr += f'\n{diff_str[0]}'
  printstr += f'\n{rel_str[0]}'
  printout.write(printstr)
  print(printstr)
  printout.close()
  
  
def split_date_time(data):
  date_time = data.index[0]
  date = date_time.split()[0]
  times = []
  for item in data.index:
    date_time = item.split()
    times.append(date_time[1])
  return(date, times)

  
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
  
  
def plot_timeseries(dataATom, dataUKCA, path):
  # Works best with data from one flight, and one field.
  date, times = split_date_time(dataUKCA)
  title = make_title(dataATom.name)
  label = make_sentence(dataATom.name)
  name = make_filename(title)
  plt.figure()
  plt.title(f'{title} ALONG FLIGHT PROGRESSION, {date}')
  x = [t[0:5] for t in times] # Cut the extra 00s for seconds off the time labels.
  y1 = dataATom
  y2 = dataUKCA
  plt.plot(x, y1, label='ATom')
  plt.plot(x, y2, label='UKCA')
  plt.xlabel('time')
  plt.ylabel(label)
  plt.legend()
  plt.savefig(f'{path}/{name}_{date}_ts.png')
  plt.show()    
  
  
def plot_corr(path, dataATom, dataUKCA, other=None, remove_null=False, remove_zero=False):
  # Works with data for one field.
  # other: another field to show as colour of dots, e.g. latitude.
  if remove_null or remove_zero:
    dataATom, dataUKCA, other = remove(dataATom, dataUKCA, other, remove_null, remove_zero)
  title = make_title(dataATom.name)
  name = make_filename(title)
  label = make_sentence(dataATom.name)
  plt.figure()
  r2 = round(r2_score(dataUKCA, dataATom), 2)
  plt.figtext(0.25, 0.75, f'r\u00b2 = {r2}')
  x = dataUKCA
  y = dataATom
  plt.scatter(x, y, c=other)
  # Make sure the axes are on the same scale for easy viewing, 
  # unless the scales are too different to see.
  #if abs(np.mean(rel_diff_no_zero(dataATom, dataUKCA))) < 50:
  if not too_diff(dataATom, dataUKCA):
    low = min([min(dataATom), min(dataUKCA)])
    high = max([max(dataATom), max(dataUKCA)])
    ext = (high - low)/20 # +5% each end to display all points clearly.
    plt.xlim(low-ext, high+ext)
    plt.ylim(low-ext, high+ext) 
  # Line of best fit.
  plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), c='black', linestyle='dotted')
  #line_eqn = get_line_eqn(x, y)
  # Display it on chart with R2 score.
  #plt.text(-5, 60, f'r\u00b2 = {r2}\n{line_eqn}')
  plt.xlabel(f'UKCA {label}') 
  plt.ylabel(f'ATom {label}')
  if other is not None:
    plt.colorbar(label=make_sentence(other.name))
  # The title depends on whether we're looking at all flights or 1 flight.
  if len(dataATom) < 24:
    date, _ = split_date_time(dataATom)
    plt.title(f'CORRELATION OF {title} FROM ATOM AND UKCA, {date}') 
    path = f'{path}/{name}_{date}_corr.png'
  else:
    plt.title(f'CORRELATION OF {title} FROM ATOM AND UKCA') 
    path = f'{path}/{name}_corr.png'
  plt.savefig(path)
  plt.show() 
  

def plot_diff(dataATom, dataUKCA, path):
  # Works best with data from all times and all flights at once, and one field.
  title = make_title(dataATom.name)
  name = make_filename(title)
  plt.figure()
  plt.title(f'UKCA DIFFERENCE TO ATOM FOR {title}')
  diff = dataUKCA - dataATom
  # Relative difference. 
  rel_diff = rel_diff_no_zero(dataATom, dataUKCA)
  plt.hist(rel_diff, bins=50, density=False) # Change denstiy to True to put % of data points on y axis.
  plt.xlabel('% difference')
  plt.ylabel('Number of data points')
  plt.savefig(f'{path}/{name}_diff.png')
  plt.show()
  
  
def plot_data(dataATom, dataUKCA, path, remove_zero=False):
  # Works best with data from all times and all flights at once, and one field.
  if remove_zero:
    dataATom, dataUKCA, _ = remove(dataATom, dataUKCA, zero=True)
  title = make_title(dataATom.name)
  name = make_filename(title)
  plt.figure()
  plt.title(f'UKCA AND ATOM DATA FOR {title}')
  # If they are too different, show their logs instead.
  #if abs(np.mean(rel_diff_no_zero(dataATom, dataUKCA))) > 50:
  if too_diff(dataATom, dataUKCA):
    dataATom = np.log(dataATom[np.nonzero(dataATom)[0]])
    dataUKCA = np.log(dataUKCA[np.nonzero(dataUKCA)[0]])
    xlabel = f'Log of {name} non-zero values'
  else:
    xlabel = name
  plt.hist(dataATom, bins=50, histtype='step', density=False, label='ATom') # Change denstiy to True to put % of data points on y axis.
  plt.hist(dataUKCA, bins=50, histtype='step', density=False, label='UKCA') 
  plt.xlabel(xlabel)
  plt.ylabel('Number of data points')
  plt.legend()
  plt.savefig(f'{path}/{name}_data.png')
  plt.show()
  
  
def plot_location(data1, data2, path):
  # Works best with data from one flight.
  date, _ = split_date_time(data1)
  fig = pylab.figure(figsize=(8,4))
  ax = pylab.axes(projection=ccrs.PlateCarree(central_longitude=310.0))
  ax.stock_img()
  pylab.title(f'Locations of selected flight path points, {date}')
  alt = data1['ALTITUDE m']
  x = data1['LONGITUDE']
  y = data1['LATITUDE']
  pylab.scatter(x+50, y, s=70, c=alt, cmap='Reds', marker='_', label='ATom')
  alt = data2['ALTITUDE m']
  x = data2['LONGITUDE']
  y = data2['LATITUDE']
  pylab.scatter(x+50, y, s=70, marker='|', c=alt, cmap='Reds', label='UKCA')
  pylab.legend(labelcolor='white', facecolor='black')
  pylab.colorbar(label='Altitude / m', shrink=0.8)
  pylab.tight_layout()
  pylab.savefig(f'{path}/{date}_loc.png')
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

for field in ATom_all.columns:
  
  # Test
  field = 'JO3 O2 O1D'

  ATom_field = ATom_all[field]
  UKCA_field = UKCA_all[field]
  #diffs(ATom_field, UKCA_field, 'ATom', 'UKCA', out_dir)
  plot_data(ATom_field, UKCA_field, out_dir, True)
  #plot_diff(ATom_field, UKCA_field, out_dir)
  plot_corr(out_dir, ATom_field, UKCA_field, remove_null=True, remove_zero=True)
  
  # End of test
  exit()
  
  for ATom_day_file in ATom_daily_files:
    ATom_day = pd.read_csv(ATom_day_file, index_col=0)
    # There are a lot of flights. Just look at the longest ones.
    if len(ATom_day) >= 10:
      date, _ = split_date_time(ATom_day)
      UKCA_day_file = f'{UKCA_dir}/UKCA_hourly_{date}.csv'
      UKCA_day = pd.read_csv(UKCA_day_file, index_col=0)
      plot_location(ATom_day, UKCA_day, out_dir)
      plot_timeseries(ATom_day[field], UKCA_day[field], out_dir)
      plot_corr(out_dir, ATom_day[field], UKCA_day[field], UKCA_day['LATITUDE'], remove_null=True)  
