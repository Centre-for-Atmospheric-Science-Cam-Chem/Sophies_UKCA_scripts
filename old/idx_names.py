import cf
import numpy as np
import prediction_fns_numpy as fns


def get_name(row, day, names):
  # Find the name of a row.
  # row: a 1d npy array from an input set.
  # day: the 2d npy array containing all the data.
  # names: the array of processed metadata of row names.
  # Get the index of the row in the full dataset.
  if np.all(row == 0):
    name = 'empty'
  else:
    for i in range(len(day)):
      each_row = day[i]
      if np.all(each_row == row):
        idx = i
        break
    # Match that index to the line of text in the metadata. 
    name = names[i][1]
  return(name)


dir_path = '/scratch/st838/netscratch/ukca_npy'
npy_file = f'{dir_path}/20150715.npy'
pp_file = f'{dir_path}/cy731a.pl20150701.pp'
name_file = f'{dir_path}/idx_names'

day_np = np.load(npy_file) 
day_cf = cf.read(pp_file)
idx_names = fns.get_idx_names(name_file)

'''
print('PP file:')
for i in range(len(day_cf)):
  print(i)
  field = day_cf[i]
  print(field.identity(), field.long_name)

print('Npy file:')
for i in range(len(day_np)):
  print(i)
  print(idx_names[i])
  field = day_np[i]
  name = get_name(field, day_np, idx_names)
  print(name)
  print()
''' 
 
for i in range(len(day_np)):
  if i > 6:
    exit()
  name = idx_names[i][1]
  code = name.split()[0]
  print(code)
  field_cf = day_cf.select(code)
  if len(field_cf) != 0: 
    field_cf = field_cf[0]
    field_cf = field_cf.array()
    field_np = day_np[i]
    print(field_np[0], field_np[-1])
    print(field_cf[0][0][0][0], field_cf[-1][-1][-1][-1])
  print()
  
