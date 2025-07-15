'''
Name: Sophie Turner.
Date: 19/1/2023.
Contact: st838@cam.ac.uk
Functions which are used by prediction scripts.
Files are located at scratch/st838/netscratch.
'''
# module load anaconda/python3/2022.05
# conda activate /home/st838/nethome/condaenv
# Tell this script to run with the currently active Python environment, not the computer's local versions. 
#!/usr/bin/env python

import math
import glob
import psutil
import warnings
import numpy as np
from PIL import Image
import constants as con
import cartopy.crs as ccrs
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_percentage_error, r2_score 
  

# Metadata functions.

def get_idx_names(name_file):
  '''Names of the fields, taken from metadata.
  name_file: file path containing array of fields' indices and names.
  Returns a list of indices and names.'''
  lines = open(name_file, 'r')
  idx_names = []
  for line in lines: 
    words = line.replace('\n', '')
    words = words.split(maxsplit=1) 
    lst = [words[0], words[1]]
    idx_names.append(lst)
  return(idx_names)  
  
  
# Data preprocessing functions.  
  
def collate(training_data_path, input_files):
  '''Collate the data from multiple files, if it hasn't already been done.
  training_data_path: file path of where to save the combined data.
  input_files: list of training data file paths.'''
  if os.path.exists(training_data_path):
    days = np.load(training_data_path)
  else:
    # Memory usage limit.
    max_mem = 75 # Percentage.
    days = np.empty((con.n_fields, 0), dtype=np.float32)
    for i in range(len(input_files)):
      print(f'Adding day {i+1} to array.')
      start = time.time()
      npy_file = input_files[i]
      data = np.load(npy_file)
      days = np.append(days, data, axis=1)
      mem_used = round(psutil.virtual_memory().percent, 1)
      if mem_used > max_mem:
        sys.exit(f'The maximum memory usage limit of {max_mem}% was exceeded with a {i+1} day array of size {days.shape}.')
      end = time.time()
      seconds = round(end - start)
      gb = round(psutil.virtual_memory().used / 1000000000, 3)
      print(f'That day took {seconds} seconds.')
      print(f'Size of array: {days.shape}.')
      print(f'Memory used so far: {gb} GB.')
      print(f'Memory usage at {mem_used}%.')
    # Save the np array for training data for re-use.
    # The .npy should be deleted after final use to free up space as it is very large.
    print('Saving .npy.')
    start = time.time()
    np.save(training_data_path, days)
    end = time.time()
    seconds = round(end - start)
    print(f'That took {seconds} seconds.')
  # Return one np array containing data from multiple files.
  return(days)


def find_cloud_end(data):
  '''Find out at what level there are no more clouds.
  data: np array of the contents of an entire .npy file of data, of shape (features, samples).     
  '''
  # For each grid & time point...
  n_points = len(data[0])
  # First altitude value.
  first_alt = data[1,0] 
  # Stride = lats x lons to check each altitude.
  stride = 144 * 192 
  j = 0
  for i in range(0, n_points, stride):      
    j += 1
    alt = data[1,i]
    # The altitudes are repeated in the data structure due to flattening. Don't process same ones again.
    if i != 0 and alt == first_alt:
      break
    # Indices of all data points at this altitude.  
    layer = np.where(data[1] == alt)  
    # Total cloud in this layer.
    cloud_in_layer = np.sum(data[con.cloud, layer])
    if cloud_in_layer == 0:
      break
  return(alt)    

  
def sum_cloud(data, out_path=None):
  '''Sum the cloud in the column above each grid box and include that as a feature in training data.
  data: np array of the contents of an entire .npy file of data, of shape (features, samples).   
  '''
  print('Summing cloud columns. This will take around half an hour or even longer.')
  # Check at which altitude clouds stop.
  clear = find_cloud_end(data)
  # Make a new array for cloud col.
  col = np.zeros(data.shape[1], dtype=np.float32)   
  # For each grid & time point...
  n_points = len(data[0]) 
  for i in range(n_points):   
    # Get the time, altitude, latitude and longitude.
    alt = data[con.alt,i]
    # Ignore any altitudes which are too high for clouds.
    if alt >= clear:
      continue
    lat = data[con.lat,i]
    lon = data[con.lon,i]
    dt = data[con.days,i] # date-time
    cloud = data[con.cloud,i]
    # Search only grid boxes after this one, not before.
    # Stride = lats x lons.
    stride = 144 * 192      
    for j in range(i, n_points, stride):
      alt_j = data[con.alt,j]
      # Ignore any altitudes which are too high for clouds.
      if alt_j >= clear:
        break
      lat_j = data[con.lat,j]
      lon_j = data[con.lon,j]
      dt_j = data[con.days,j] # date-time
      # Get all the grid boxes where time, lat & lon are the same AND alt > this alt.
      if dt_j == dt and alt_j > alt and lat_j == lat and lon_j == lon:
        # Sum the cloud of those grid boxes.
        cloud_j = data[con.cloud,j]
        cloud += cloud_j
    # Add this number to the new feature at this grid box.
    col[i] = cloud
  # Add the new array as a feature in the dataset, in a new column after the current cloud one.
  data = np.insert(data, con.cloud+1, col, axis=0)       
  # Save the new dataset.
  if out_path is not None:
    print('Saving new dataset.')
    np.save(out_path, data)
  # Return the new dataset.
  return(data)
  

def shape(rows):
  '''Make single input fields 2d so that they fit with other parameters going into the ML functions.
  rows: np array of fields chosen for training or testing.
  '''
  if rows.ndim == 1:
    rows = np.vstack((rows, ))
  return(rows) 
  
  
def remove_all_zero(data):
  '''Remove all-zero fields from dataset.'''
  removes = []
  for i in range(len(data)):
    field = data[i]
    if np.all(field == 0):
      removes.append(i)
  data = np.delete(data, removes, axis=0)
  return(data)
  
  
def all_zeros(field):
  '''Check if a field contains only zeros.'''
  zeros = False
  if np.all(field == 0):
    zeros = True
  return(zeros)
  
 
def get_name(idx, idx_names):
  '''Get the name of a field from its index number.'''
  for idx_name in idx_names:
    if idx_name[0] == idx:
      name = idx_name[1].split()[1]
      name = name.split('_')[0]
      return(name)
 
 
def day_trop(data):
  '''Remove night and upper stratosphere were Fast-J shouldn't be used.
  data: 2d numpy array of flattened UM output data, of shape (features, samples).
  '''
  # Remove night.
  data = data[:, np.where(data[con.down_sw_flux] > 0)].squeeze()
  # Remove upper portion where Fast-J might not work.
  data = data[:, np.where(data[con.pressure] > 20)].squeeze()  
  return(data) 
      
  
def split_pressure(data):
  '''Split the UKCA dataset into two: one above and one below the Fast-J cutoff, using pressure.
  Data: 2D np array of whole UKCA dataset, of shape (features, samples).
  '''
  cutoff = 20 # Pascals. 
  bottom = data[:, np.where(data[con.pressure] > cutoff)].squeeze()
  top = data[:, np.where(data[con.pressure] < cutoff)].squeeze()
  return(bottom, top)
  
  
def sample(data, size=None):
  '''Make a smaller dataset by randomly sampling the data, uniformly.
  Choose a big enough size to capture sufficient tropospheric density.
  To do: use a better function than random uniform sampling.
  data: np array of dataset, 1D or 2D of shape (features, samples).
  size: number of data points desired. Leave empty for auto (10%).
  '''
  if data.ndim == 1:
    length = len(data)
  else: 
    length = len(data[0])
  if size is None:
    size = round(length / 10) # 10%.
  i = con.rng.integers(0, length, size, dtype=np.int32)
  if data.ndim == 1:
    data = data[i] 
  else:
    data = data[:, i] 
  return(data)
  
  
def in_out_swap(data, i_inputs=con.phys_main, i_targets=con.J_core):
  '''Select inputs and targets for using ML and swap dimensions over to make data compatible with random forest shape.
  data: 2D numpy array of flattened UKCA output data, of shape (features, samples).
  i_inputs, i_targets: optional 1D list of ints, indices of input/target features. 
  Returns data in shape (samples, features).
  '''
  # Input data.
  inputs = data[i_inputs]
  if inputs.ndim == 1:
    inputs = inputs.reshape(1, -1) 
  inputs = np.swapaxes(inputs, 0, 1)
  print('Inputs:', inputs.shape)
  # Target data.
  targets = data[i_targets]
  if targets.ndim > 1:
    targets = np.swapaxes(targets, 0, 1) 
  print('Targets:', targets.shape)
  return(inputs, targets)
  
  
def tts(inputs, targets, test_size=0.1):
  '''Random but sorted train test split function which returns indices for coords.
  inputs: 2d np array of input data, of shape (samples, features).
  targets: 2d np array of target data, of shape (samples, features).
  test_size: float out of 1, proportion of data to use for test set. 
  Returns TTS data and indices of split in shape (samples, features).
  '''
  # Find length of data and size of test set.  
  len_all = len(inputs)
  len_test = round(len_all * test_size)
  i_all = np.arange(0, len_all, dtype=int)
  # Generate random, unique indices for selecting test set.
  i_test = con.rng.choice(i_all, len_test, replace=False) 
  # Put them back into the right order.
  i_test = np.sort(i_test)
  # Make test set.
  in_test = inputs[i_test]
  out_test = targets[i_test]
  # Remove these from the training set.
  in_train = np.delete(inputs, i_test, axis=0)
  out_train = np.delete(targets, i_test, axis=0)  
  # Return train and test datasets and their indices from the full dataset.
  return(in_train, in_test, out_train, out_test, i_test) 
  
  
def lvl_to_alt(orography: float, include_surface: bool = False) -> list:
    '''Function to calculate model level heights above ground for a single vertical column. 
    Based on code written by Rob Walters.
    orography: float of ground height for a vertical column, usually from STASH section 0, item 33.
    include_surface: optional bool of whether or not to include the lowest model level. Defaults to False.
    Returns alts: list of altitudes of model levels. 
    '''
    # Check sigma.
    eta_first_constant_rho = con.levels['eta_rho'][
        con.levels['first_constant_r_rho_level'] - 1]
    alts = []
    if include_surface:
        starting_index = 0
    else:
        starting_index = 1
    for i in range(starting_index, len(con.levels['eta_rho'])): 
        eta = con.levels['eta_rho'][i]
        if i >= con.levels['first_constant_r_rho_level'] :
            # Not terrain following.
            sigma = 0
        else:
            # Scaling factor for topography following coords.
            sigma = (1.0 - (eta / eta_first_constant_rho))**2
        # With no terrain use eta.
        delta = eta * con.levels['z_top_of_model']
        alts.append(delta + (sigma * orography))
    return(alts)   
 
  
# ML functions.  
  
def train(in_train, out_train):
  '''Set up simple model.'''
  model = linear_model.LinearRegression()
  model.fit(in_train, out_train)
  return(model)


def sMAPE(out_test, out_pred):
  '''Compute the symmetric mean absolute percentage error.
  Like MAPE but handles values close to zero better.
  test: array of test targets. pred: array of prediction outputs. 
  '''
  warnings.filterwarnings('ignore') 
  n = len(out_test)
  diffs = 0
  for i in range(n):
    diffs += (abs(out_test[i] - out_pred[i])) / (out_test[i] + out_pred[i])
  smape = (100 / n) * diffs 
  return(smape)
  
  
def metrics_2d(out_test, out_pred):
  '''Get metrics over 2 dimensions for metrics which can only be used on 1 dim.
  out_test, out_pred: 2D arrays.
  '''
  n = len(out_test[0])
  maxes = [] # Array of max errors for each target.
  smape = 0 # Average percentage error.
  for target in range(n):
    target_test = out_test[:,target].squeeze()
    target_pred = out_pred[:,target].squeeze()
    maxe = max_error(target_test, target_pred)
    maxes.append(maxe)
    smape += sMAPE(target_test, target_pred)
  smape = smape / n
  return(maxes, smape)
  

def test(model, in_test, out_test):
  # Try it out.
  out_pred = model.predict(in_test)
  # See how well it did.
  if out_test.ndim == 1:
    maxe = max_error(out_test, out_pred)
    smape = sMAPE(out_test, out_pred)
  else:
    maxe, smape = metrics_2d(out_test, out_pred)
  mse = mean_squared_error(out_test, out_pred)
  mape = mean_absolute_percentage_error(out_test, out_pred)
  r2 = round(r2_score(out_test, out_pred), 3)
  return(out_pred, maxe, mse, mape, smape, r2) 
  
  
def only_range(inputs, targets, targets_all, bottom=0, top=1, keep=True, min_samples=10):
  '''Only use a certain range of values of J rates.
  Call just before fns.test() and pass only test sets if you want model trained on all data but only tested on smallest.
  inputs: array of inputs (in_test for test-only).
  targets: array of targets (out_test for test-only).  
  targets_all: array of all targets in data, used for the test-only option. If using all data, just pass the targets for both target params.
  bottom: chosen proportion of data for lowest value, 0 to 1, inclusive.
  top: chosen proportion of data for highest value, 0 to 1, inclusive.
  min_samples: minimum number of samples a J rate needs for it to be included. 
  keep: whether to keep the selected portion and remove the rest (True), or remove the selected portion and keep the rest (False). 
  '''
  assert bottom < top
  if targets.ndim == 1:
    top_all = max(targets_all)
    bottom_new = top_all * bottom
    top_new = top_all * top
    i_range = np.where((targets >= bottom_new) & (targets <= top_new))
    # Reduce the data to these indices.
    if keep:
      inputs = inputs[i_range]
      targets = targets[i_range]
    else:
      inputs = np.delete(inputs, i_range, axis=0)
      targets = np.delete(targets, i_range)
  else:
    deletes = np.array([], dtype=int)
    # For each target feature...
    for i in range(len(targets_all[0])):
      target = targets_all[:, i]
      # Get the index.
      top_all = max(target)
      bottom_new = top_all * bottom
      top_new = top_all * top
      i_range = np.where((targets[:, i] >= bottom_new) & (targets[:, i] <= top_new)) 
      if keep:
        # Sometimes there are no data in this region for some J rates.
        if len(i_range[0]) < min_samples:
          deletes = np.append(deletes, i)
        else:
          # Reduce the data to these indices.
          inputs = inputs[i_range]
          targets = targets[i_range]
      else:
        inputs = np.delete(inputs, i_range, axis=0)
        targets = np.delete(targets, i_range, axis=0)
    if keep:
      # Remove these J rates from the selection.
      targets = np.delete(targets, deletes, axis=1)
  return(inputs, targets, deletes)   
  
  
# Results & postprocessing functions.  

def load_model_data(mod_name='rf'):
  '''Loads and returns input, target and prediction sample data from a trained ML model.
  mod_name: string, name of ML model and its folder.
  '''
  # File paths.
  mod_path = f'{paths.mod}/{mod_name}/{mod_name}'
  inputs_path = f'{mod_path}_inputs.npy' 
  targets_path = f'{mod_path}_targets.npy'
  preds_path = f'{mod_path}_pred.npy'
  # Load data.
  print('\nLoading data.')
  inputs = np.load(inputs_path)
  targets = np.load(targets_path)
  preds = np.load(preds_path)
  print('Inputs:', inputs.shape)
  print('Targets:', targets.shape)
  print('Preds:', preds.shape)
  return(inputs, targets, preds)
  
  
def make_cols_map(inputs, targets, preds, j_idx=None, out_path=None):
  '''Make an array of column performance.
  inputs, targets, preds: 2d numpy arrays of ML model test data, of shape(samples, features).
  j_idx: int or None, index in target data of which J rate to look at. If None, plot avg of all.
  out_path: string or None. File path to save map as.
  Returns the map as a 2d numpy array of lat, lon, r2, diff for each col.
  '''
  warnings.filterwarnings('ignore') # R2 score on 1 datapoint.
  # Get the unique lat & lon values.
  lats = inputs[:, con.lat]
  lats = np.unique(lats)
  lons = inputs[:, con.lon]
  lons = np.unique(lons)
  # Make a 2d list of coords, R2 scores and % differences for each lat and lon point.
  grid = []
  # For each lat...
  for i in range(len(lats)):
    lat = lats[i]
    # And each lon...
    for j in range(len(lons)):
      lon = lons[j]
      # Get the indices of all the data in that column.
      idx = np.where((inputs[:, con.lat] == lat) & (inputs[:, con.lon] == lon))[0]
      # Get all the target J rates in that column.
      target = targets[idx, j_idx].squeeze()     
      # There might not be a data point at this location. Skip if so.
      if not np.any(target) or target.ndim == 0:
        continue
      # Get the predicted J rates in the column.
      pred = preds[idx, j_idx].squeeze() 
      # Get the R2 score and % diff.
      r2 = r2_score(target, pred)
      diff = np.nan_to_num(((np.mean(pred) - np.mean(target)) / np.mean(target)) * 100, posinf=0, neginf=0)
      # Save the R2 scores in a 2d list for every lat & lon.
      grid.append([lat, lon, r2, diff])     
  grid = np.array(grid)
  #print(grid.shape)
  if out_path is not None:
    # Save the array because it takes ages to make.
    np.save(out_path, grid)
  # Returns the map as a 2d numpy array of lat, lon, r2, diff for each col.
  return(grid)
  

# Plot functions.

def shrink(out_test, out_pred):
  '''Don't plot an unnecessary number of data points i.e. >10000.'''  
  # Make them the right shape.
  out_test = out_test.squeeze()
  out_pred = out_pred.squeeze()
  length = len(out_pred)
  if length > 10000:
    # Plotting this many datapoints is excessive and costly. Reduce it to 10000.
    idxs = np.int16(np.linspace(0, length-1, 10000))
    out_test = out_test[idxs]
    out_pred = out_pred[idxs]
    del(idxs)
    # Choose opacity of points.
    alpha = 0.05
  elif length > 100:
    alpha = 0.1
  else:
    alpha = 1
  return(out_test, out_pred, alpha)


def force_axes():
  '''Make plot axes exactly the same.'''
  plt.axis('square')
  xticks, xlabels = plt.xticks()
  yticks, ylabels = plt.yticks()
  plt.axis('auto')
  if len(yticks) > len(xticks):
    tix = yticks
    lab = ylabels
  else:
    tix = xticks
    lab = xlabels
  plt.xticks(ticks=tix, labels=lab)
  plt.yticks(ticks=tix, labels=lab) 
  

def line(xdata, ydata, xdata2=None, ydata2=None, title='', xlab='', ylab='', leg1='', leg2=''):
  '''Show a line plot of 2 vars.
  xdata, ydata: arrays to plot.
  xdata2, ydata2: optional 2nd line to plot.
  xlab, ylab: optional axis text.
  leg1, leg2: optional legend labels if using 2 lines.  
  '''
  plt.plot(xdata, ydata, label=leg1)
  if xdata2 is not None:
    plt.plot(xdata2, ydata2, label=leg2)
    plt.legend()
  plt.title(title)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.show()
  plt.close()
  
  
def show(out_test, out_pred, maxe=None, mse=None, mape=None, smape=None, r2=None, j='', t=None):
  r2 = round(r2, 3)
  if maxe is not None: print(f'MaxE = {maxe}') 
  if mse is not None: print(f'MSE = {mse}') 
  if mape is not None: print(f'MAPE = {mape}') 
  if smape is not None: print(f'SMAPE = {smape}')
  if t is None: t = j
  # Don't plot >10000 points.
  out_test, out_pred, a = shrink(out_test, out_pred)
  plt.scatter(out_test, out_pred, alpha=a)
  # Make a 1:1 reference line.
  plt.plot(out_test, out_test, linestyle=':', color='grey', alpha=0.5)
  # Force axes to be identical.
  force_axes()
  if r2 is not None:
    print(f'Coefficient of determination = {r2}')
    plt.title(f'{t} J rates. {con.r2}={r2}')
  plt.xlabel(f'{j} J rate from Fast-J / {con.pers}')
  plt.ylabel(f'{j} J rate from random forest / {con.pers}')
  plt.show() 
  plt.close()
  
  
def show_diff_map(fullname, grid, r2, extra_info=None, out_path=None):
  '''Show a map of the column % diffs for a specific J rate.
  fullname: string, fully formatted name of J rate.
  grid: 2d numpy array of lat, lon, r2, diff for each column.
  r2: number, overall R2 score of whole grid at this timestep.
  extra_info: string, any more text needed in plot title.
  out_path: string, file path to save fig as.
  '''
  # Show plot for this J rate and ts. 
  cmap = con.cmap_diff
  # Clip the bounds to +- 20% to remove ridiculous outliers and simplify the plot visuals.
  vmin, vmax = -20, 20
  x = grid[:, 1]
  y = grid[:, 0]
  c = grid[:, 3]
  # Set the fig to a consistent size.
  plt.figure(figsize=(10,7.5))
  # Plot the metrics by lat & lon coords on the cartopy mollweide map.
  ax = plt.axes(projection=ccrs.Mollweide()) 
  plt.title(f'Columns of {fullname} photolysis rates predicted by random forest {extra_info}\nOverall {con.r2} = {r2}')
  plt.scatter(x, y, c=c, s=3, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree()) 
  plt.colorbar(shrink=0.5, label=f'% difference of J rate predictions to targets in column', orientation='horizontal')
  ax.set_global()
  ax.coastlines()
  # Save or show the map.
  if out_path is None:
    plt.show()
  else:
    plt.savefig(out_path)
  plt.close() 
  
  
def show_col_orig(data, ij=78, name='O3', all_time=True):
  '''Show a column of J rate by altitude for full UKCA datasets before ML. 
  data: 2d np array of full dataset.
  ij: index of J rate to look at. 78=O3 in full dataset.
  name: name of J rate.
  all_time: whether to include all time steps (True) or plot a line of one time point (False).
  '''
  # Pick a lat-lon point and a time.
  # Cambridge at midday.
  lat, lon, hour = 51.875, 0.9375, 12
  # Choose whether to specify the time point.
  if not all_time:
    # Find the first occurance of this time so that we only select one day.
    data = data[:, np.where(data[con.hour] == hour)].squeeze()
    dt0 = np.unique(data[con.days])[0]
    data = data[:, np.where(data[con.days] == dt0)].squeeze()
  # Fetch this grid col's data.
  data = data[:, np.where((data[con.lat] == lat) & (data[con.lon] == lon))].squeeze() 
  # Pick out the J rate and altitude.
  j = data[ij]
  alts = data[con.alt]
  # Convert to km.
  alts = alts * 85
  # Plot the J rate by altitude in that column.
  if all_time:
    plt.scatter(j, alts, label=f'J{name} from UKCA', alpha=0.2)
    plt.title('UKCA column ozone J rates over Cambridge in 2015')
  else:
    plt.plot(j, alts, label=f'J{name} from UKCA')
    plt.title('UKCA column ozone J rates over Cambridge at midday on 15/1/2015')
  plt.xlabel(f'{con.rxnO3} J rate')
  plt.ylabel('Altitude / km')
  plt.show()
  plt.close()
 
 
def col(data, coords, lat, lon, hour, ij):
  '''Show a column of J rate by altitude to compare targets and predictions. 
  data: 2d np array dataset e.g. full UKCA data or ML targets.
  lat, lon: latitude and longitude of chosen column.
  hour: hour of day of chosen sample or None if plotting all time points.
  ij: index of J rate to look at.
  '''
  # Stick the coords on for easier processing.
  if data.ndim == 1:
    data = np.expand_dims(data, 1)
  data = np.append(coords, data, axis=1)
  if hour is not None:
    # Find the first occurance of this time so that we only select one day.
    data = data[np.where(data[:, con.hour] == hour)].squeeze()
    dt0 = np.unique(data[:, con.days])[0]
    data = data[np.where(data[:, con.days] == dt0)].squeeze()
  # Fetch this grid col's data.
  data = data[np.where((data[:, con.lat] == lat) & (data[:, con.lon] == lon))].squeeze() 
  # Pick out the J rate and altitude.
  ij += 5 # Account for added coords.
  j = data[:, ij]
  alts = data[:, con.alt]
  alts = alts * 85 # Convert proportion to km.
  return(j, alts)
  
  
def show_col(out_test, out_pred, coords, ij, name='O3', all_time=True):
  '''Show a column of J rate by altitude. 
  out_test: 2d np array dataset of targets.
  out_pred: 2d np array dataset of predictions.
  coords: 2d np array of time & space co-ordinates for datasets.
  ij: index of J rate to look at.
  name: name of J rate.
  all_time: whether to include all time steps (True) or plot a line of one time point (False).
  '''
  # Pick a lat-lon point and a time.
  lat, lon, hour = 51.875, 0.9375, 12 # Cambridge at midday.
  # Swap the dimensions to make them compatible with sklearn.
  coords = np.swapaxes(coords, 0, 1)
  # Choose whether to specify the time point.
  if all_time:
    hour = None
  j1, alts = col(out_test, coords, lat, lon, hour, ij)
  j2, alts = col(out_pred, coords, lat, lon, hour, ij)  
  # Plot the J rate by altitude in that column.
  if all_time:
    plt.scatter(j1, alts, label=f'J{name} from UKCA', marker='|', s=50, alpha=0.5)
    plt.scatter(j2, alts, label=f'J{name} from random forest', marker='_', s=50, alpha=0.5)
    plt.title('UKCA J rates, column over Cambridge in 2015')
  else:
    plt.plot(j1, alts, label=f'J{name} from UKCA')
    plt.plot(j2, alts, label=f'J{name} from random forest') 
    plt.title(f'UKCA J{name}, column over Cambridge at midday on 15/7/2015')
  plt.legend()    
  plt.xlabel(f'J{name} / {con.pers}')
  plt.ylabel('Altitude / km')
  plt.show()
  # Column percentage difference plot.
  diff = np.nan_to_num(((j2 - j1) / j1) * 100, posinf=0, neginf=0) # Accounts for div by zero.
  if all_time:
    plt.hist(diff, bins=100)
  else:
    plt.plot(diff, alts)
    plt.axvline(linestyle=':')
  plt.title(f'Column % difference of J{name} predictions to UKCA outputs')
  plt.xlabel('% difference')
  plt.ylabel('Altitude / km')
  plt.show()
  plt.close()    
  
  
def make_gif(dir_path, ext='png', gif_name='GIF', ms=500):
  '''Makes a GIF file from a directory containing the desired images, and saves it in that directory. Uses all images in the dir with the specified extension.
  dir_path: string, file path to the desired directory.
  ext: optional string, file type extension of desired images.
  gif_name: optional string, file name of resultant GIF. 
  ms: int, milliseconds spent on each frame.
  '''
  img_paths = sorted(glob.glob(f'{dir_path}/*.{ext}'))
  frames = [Image.open(image) for image in img_paths]
  frame1 = frames[0]
  frame1.save(f'{dir_path}/{gif_name}.gif', format='GIF', append_images=frames, save_all=True, duration=ms, loop=0) # Half a second per image.   
 

# Ancilliary functions.

def format_date(date):
  '''Turn date into dd/mm/yyyy format.
  date: string of format yyyymmdd.
  '''
  return(f'{date[6:]}/{date[4:6]}/{date[0:4]}')
 

def mem():
  '''Memory usage.'''
  GB = 1000000000
  mem_used = psutil.virtual_memory().used / GB
  print(f'Current memory usage: {mem_used} GB.')
