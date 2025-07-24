'''
Name: Sophie Turner.
Date: 29/11/2024.
Contact: st838@cam.ac.uk.
Test random forest on a column of input data.
'''

import re
import glob
import joblib
import numpy as np
from math import pi
import file_paths as paths
import constants as con
import prediction_fns as fns
from sklearn.metrics import mean_absolute_percentage_error, r2_score 

# File paths.
date = '20150621'
date_path = f'{paths.npy}/col{date}'
col_paths = glob.glob(f'{date_path}*.npy')
all_cols = glob.glob(f'{paths.npy}/col2015*.npy')
mod_name = 'rf'
mod_path = f'{paths.mod}/{mod_name}/{mod_name}'
model_path = f'{mod_path}.pkl'
in_scale_path = f'{mod_path}_in_scaler.pkl'
out_scale_path = f'{mod_path}_out_scaler.pkl'
idx_names_path = f'{paths.npy}/idx_names'

for col_path in all_cols:  
  print('\nLoading test data.')
  data = np.load(col_path)
  print(data.shape)
  
  # See where and when the column is.
  cols = [con.np12, con.cam12, con.gg12, con.ac12, con.sp12]
  date = re.search(f'{paths.npy}/col(.*?)_', col_path)
  date = date.group(1)
  col_name = re.search(f'{date}_(.*?).npy', col_path)
  col_name = col_name.group(1)
  text = fns.format_date(date)
  for col in cols:
    if col_name == col[4]:
      text = f'{col[3]} on {text}'
      print()
      print(text)
      break
  
  # Look at the sunlight flux in the column.
  flux = data[9] + data[10]
  fns.line(flux, data[1]*85, title=text, xlab=f'Sunlight flux / {con.Wperm2}', ylab='Altitude / km')

  # Load ML model and its scalers.
  model = joblib.load(model_path)
  #in_scale = joblib.load(in_scale_path)
  #out_scale = joblib.load(out_scale_path)
  
  # Standardise inputs. Needs to be the same as the trained model.
  inputs = data[con.phys_no_o3]
  # Get altitudes in km.
  alts = inputs[1] * 85
  # Get SZA in degrees.
  sza = np.arccos(inputs[8]) * 180 / pi
  # Make the inputs the right shape.
  inputs = np.swapaxes(inputs, 0, 1)
  #inputs = in_scale.transform(inputs)
  
  # Use the ML on these data. Needs to be the same as the trained model.
  targets = data[con.J_core]
  # Make them the right shape.
  targets = np.swapaxes(targets, 0, 1)
  
  preds = model.predict(inputs)

  # Destandardise the predictions, except NO3, which was not standardised.
  #NO3 = preds[:, 11].copy()
  #preds = out_scale.inverse_transform(preds)
  #preds[:, 11] = NO3

  # See how it did overall.
  r2 = round(r2_score(targets, preds), 3)
  mape = mean_absolute_percentage_error(targets, preds)
  fns.show(targets, preds, mape=mape, r2=r2, t='All')

  # Have a look at the inputs.
  input_names = ['Specific humidity', 'Cloud fraction', 'Pressure / Pa', 'Solar zenith angle / degrees', 
                f'Upward shortwave flux / {con.Wperm2}', f'Downward shortwave flux / {con.Wperm2}', 
		f'Upward longwave flux / {con.Wperm2}', f'Downward longwave flux / {con.Wperm2}', 'Temperature / K']
  for i in range(len(inputs[0]) - 5):
    if i == 8 - 5:
      field = sza
    else:
      field = inputs[:, i + 5]  
    field_name = input_names[i]
    print(f'\n{field_name}')
    print(f'Min: {np.min(field)}, max: {np.max(field)}, mean: {np.mean(field)}')
    fns.line(field, alts, title=text, xlab=field_name, ylab='Altitude / km') 
  
  # Get names of reactants.
  idx_names = fns.get_idx_names(idx_names_path)
  
  # Find the best and the worst predictions.
  bestr2 = 0.0
  worstr2 = 1.0
  lessthan0 = 0
  lessthan1 = 0
  lessthan2 = 0
  lessthan3 = 0
  lessthan4 = 0
  lessthan5 = 0
  lessthan6 = 0
  lessthan7 = 0
  lessthan8 = 0
  lessthan9 = 0
  lessthan100 = 0
  
  
  for i in range(len(targets[0])):
    ji = con.J_core[i]
    j = fns.get_name(str(ji), idx_names)
    target = targets[:, i]
    avgj = np.mean(target)
    print(f'Average J rate for {j}: {avgj}')
    
    pred = preds[:, i]
    r2 = r2_score(target, pred)
    mape = mean_absolute_percentage_error(target, pred)
    print(f'{j}: {con.r2}={round(r2, 3)}, MAPE={round(mape, 3)}, Mean J rate = {avgj}' )
    #fns.line(target, alts, pred, alts, f'{j}, {text}', f'J rate / {con.pers}', 'Altitude / km', f'{j} from UKCA', f'{j} from random forest')  
    # See if it's good or bad.
    if r2 >= 0.9: lessthan100 += 1
    elif r2 >= 0.8: lessthan9 += 1
    elif r2 >= 0.7: lessthan8 += 1
    elif r2 >= 0.6: lessthan7 += 1
    elif r2 >= 0.5: lessthan6 += 1
    elif r2 >= 0.4: lessthan5 += 1
    elif r2 >= 0.3: lessthan4 += 1
    elif r2 >= 0.2: lessthan3 += 1
    elif r2 >= 0.1: lessthan2 += 1
    elif r2 >= 0.0: lessthan1 += 1
    elif r2 <  0.0: lessthan0 += 1
    # See if it's the best or the worst.
    if r2 > bestr2:
      besti = i
      bestr2 = r2
      bestmape = mape
    if r2 < worstr2:
      worsti = i
      worstr2 = r2
      worstmape = mape
  
  # See how many were good and bad.
  print(f'\n< 0: {lessthan0}')
  print(f'0-0.1: {lessthan1}')
  print(f'0.1-0.2: {lessthan2}')
  print(f'0.2-0.3: {lessthan3}')
  print(f'0.3-0.4: {lessthan4}')
  print(f'0.4-0.5: {lessthan5}')
  print(f'0.5-0.6: {lessthan6}')
  print(f'0.6-0.7: {lessthan7}')
  print(f'0.7-0.8: {lessthan8}')
  print(f'0.8-0.9: {lessthan9}')
  print(f'0.9-1: {lessthan100}')
  
  # Reverse the standardisation of the inputs to make meaningful plots. 
  #inputs = in_scale.inverse_transform(inputs) 
 
  # Show the best.   
  ji = con.J_core[besti]
  j = fns.get_name(str(ji), idx_names)
  print(f'\nBest J rate: {j}') 
  target = targets[:, besti]
  pred = preds[:, besti]  
  # Make the r2 neat but don't imply it's perfect.
  bestr2 = round(bestr2, 3)
  if bestr2 == 1.0:
    bestr2 = 0.999
  txt = f'{j}, {text}'
  fns.show(target, pred, mape=bestmape, r2=bestr2, t=txt)
  # Plot by altitude.
  fns.line(target, alts, pred, alts, txt, f'J rate / {con.pers}', 'Altitude / km', f'{j} from UKCA', f'{j} from random forest')  

  # Show the worst.
  ji = con.J_core[worsti]
  j = fns.get_name(str(ji), idx_names)
  print(f'\nWorst J rate: {j}')
  target = targets[:, worsti]
  pred = preds[:, worsti]  
  worstr2 = round(worstr2, 3)
  txt = f'{j}, {text}'
  fns.show(target, pred, mape=worstmape, r2=worstr2, t=txt)
  # Plot by altitude.
  fns.line(target, alts, pred, alts, txt, f'J rate / {con.pers}', 'Altitude / km', f'{j} from UKCA', f'{j} from random forest')  
  
