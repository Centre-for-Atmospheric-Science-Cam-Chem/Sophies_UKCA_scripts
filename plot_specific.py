'''
Name: Sophie Turner.
Date: 12/12/2024.
Contact: st838@cam.ac.uk.
Plot binned avgs and ranges of J rate differences between target and prediction
for specific input variables and reactions.
'''

import warnings
import numpy as np
from math import pi
from scipy import stats
import matplotlib.pyplot as plt
import file_paths as paths
import constants as con
from sklearn.metrics import r2_score

# File paths.
mod_name = 'rf'
mod_path = f'{paths.mod}/{mod_name}/{mod_name}'
inputs_path = f'{mod_path}_test_inputs.npy' 
targets_path = f'{mod_path}_test_targets.npy'
preds_path = f'{mod_path}_pred.npy'

# The warnings are about invalid log10 values and don't matter.
warnings.simplefilter("ignore") 

# Load data.
print('\nLoading data.')
inputs = np.load(inputs_path)
targets = np.load(targets_path)
preds = np.load(preds_path)
print('Inputs:', inputs.shape)
print('Targets:', targets.shape)
print('Preds:', preds.shape)

# Sub-sample data.
size = len(preds)
i = con.rng.integers(0, size, round(size/100), dtype=np.int32)
inputs = inputs[i]
targets = targets[i]
preds = preds[i]
print('Inputs:', inputs.shape)
print('Targets:', targets.shape)
print('Preds:', preds.shape)
size = len(preds)

# Indices of J rates in J_core which are prone to poorer prediction performance.
# NO2, CH3COCHO, NO3, HOBr, MACR.
js = [0, 3, 11, 12, 20]  
j_names = [f'NO{con.sub2}', f'NO{con.sub3}', f'CH{con.sub3}COCHO', 'Methacrolein', 'HOBr']

# Indices of J rate groups in J_core.
#js = [11,7,14,10,16,13,6,18]
#j_names = ['NOx', 'nitrate', 'nitric acid', 'aldehyde', 'peroxide', 'hypohalous acid', f'SO{con.sub3}', 'ozone']

# Array for their % differences.
diffs = np.empty((len(js), size), np.float32)
for i in range(len(js)):
  target = targets[:, js[i]]
  pred = preds[:, js[i]]
  diff = np.nan_to_num((((pred - target) / target) * 100), posinf=0, neginf=0) # %
  #diff = np.nan_to_num(((pred - target) / target), posinf=0, neginf=0) # Proportion
  diffs[i] = diff

'''
# Get errors and differences of J rates.
size = len(preds)
mse_all = np.empty(size) # MSE for all J rates.
for i in range(size):
  mse_all[i] = mse(targets[i], preds[i])
print('MSE all:', mse_all.shape)

mse_all = mse(targets, preds) # MSE for all J rates.
mse_no2 = mse(targets_no2, preds_no2) # MSE for NO2.
mse_no3 = mse(targets_no3, preds_no3) # MSE for NO3.
mse_ch3cocho = mse(targets_ch3cocho, preds_ch3cocho)
mse_macr = mse(targets_macr, preds_macr)
mse_hobr = mse(targets_hobr, preds_hobr) 
'''
# Have a look at the inputs.
input_names = ['Hour of day', 'Altitude / km', 'Latitude / deg', 'Longitude / deg', 'Days since 1/1/2015',
               'Specific humidity', 'Cloud fraction', 'Pressure / Pa', 'Solar zenith angle / degrees', 
              f'Upward shortwave flux / {con.Wperm2}', f'Downward shortwave flux / {con.Wperm2}', 
              f'Upward longwave flux / {con.Wperm2}', f'Downward longwave flux / {con.Wperm2}', 'Temperature / K']

# Convert altitude and SZA to more interpretable units.
alt = inputs[:, 1] * 85 # km.
inputs[:, 1] = alt
sza = np.arccos(inputs[:, 8]) * 180 / pi # Degrees.
inputs[:, 8] = sza

# For each input variable...
for i in range(len(input_names)):  

  field = inputs[:, i]
  field_name = input_names[i]
  # Get input values.
  vals = np.unique(field) 
  # Make a list to store average differences, highs and lows.
  n_vals = len(vals)
  avgs = np.empty(n_vals, np.float32)
  highs = np.empty(n_vals, np.float32)
  lows = np.empty(n_vals, np.float32)
  # For each J rate...
  for j in range(len(js)):
    x = field
    diff = diffs[j]
    lab = j_names[j]
    '''
    # Cut extreme outliers off.    
    idx = []
    for diffi in range(len(diff)):
      diffv = diff[diffi]
      if diffv < -100 or diffv > 100:
        idx.append(diffi)
    diff = np.delete(diff, idx)
    x = np.delete(x, idx)  

    # Or use a confidence interval instead.
    mean = np.mean(diff)
    sem = stats.sem(diff) # Standard error of the mean.
    ci = stats.t.interval(0.95, size-1, loc=mean, scale=sem) # 95% confidence interval.
        
    # Or use standard deviations instead.
    z_scores = stats.zscore(diff)
    idx = np.where(diff[np.abs(z_scores) > 1])
    diff = np.delete(diff, idx)
    x = np.delete(x, idx) 
    '''
          
    # Binned moving average and shading.
    bins = 80 
    # Easier to just use the binning even if it isn't needed than have separate code.
    if len(vals) < bins:
      bins = len(vals)
    # Get bin size.
    window = round(len(vals) / bins)  
    # Get bounds of x.
    xmin = np.min(vals)
    xmax = np.max(vals)
    # Make an array for new x values of bins.
    x_bins = np.linspace(xmin, xmax, bins)    
    # Make an array for new y values.
    y_bins = np.empty(bins)
    # Make arrays for min and max lines for shaded area.
    low_bins, high_bins = np.empty(bins), np.empty(bins)
    # Loop through y in bin sized steps.
    for binn in range(bins): 
      # Get the indices of bin min and max in full x data.
      xidx = binn * window
      # Get the min and max x values in the bin.
      binxmin = x[xidx]
      binxmax = x[xidx + window - 1]
      # Get all the J rate differences at the input values in that bin.      
      diff_in_bin = diff[(x >= binxmin) & (x <= binxmax)] 
      
      # Sometimes there are no data in this x range.
      if len(diff_in_bin) == 0:
        avg, low, high, iqr = np.nan, np.nan, np.nan, np.nan
      else: 
        # Average them.
        avg = np.nanmedian(diff_in_bin)
        # Shade areas where points are.
        # Get upper and lower quartile y points values in the bin.
        low = np.nanpercentile(diff_in_bin, 25)
        high = np.nanpercentile(diff_in_bin, 75)
      
      # Add the average value of the bin to the new y array.
      y_bins[binn] = avg
      # Add the bottom and top lines.
      low_bins[binn] = low
      high_bins[binn] = high
      
    # Optional: remove nans from arrays to plot. Keep nans if you want to see where the missing data are.
    nans = np.isnan(y_bins)
    x_bins, y_bins, low_bins, high_bins = x_bins[~(nans)], y_bins[~(nans)], low_bins[~(nans)], high_bins[~(nans)] 	
	
    # Plot the shaded area.	
    plt.fill_between(x_bins, low_bins, high_bins, alpha=0.1)	
    # Plot the moving average.
    plt.plot(x_bins, y_bins, label=f'{lab} median difference')   
     
  # Put a zero line on the plot.
  zero = np.zeros(len(x_bins))
  plt.plot(x_bins, zero, color='grey', linestyle='dashed', alpha=0.5)     
  plt.title(f'Effect of input data on photolysis rates prone to poor predictions by a random forest')
  plt.xlabel(field_name)
  plt.ylabel('% difference between predicted and target J rate')
  plt.legend(loc='upper left', framealpha=0)
  plt.show()
  plt.close()
  
