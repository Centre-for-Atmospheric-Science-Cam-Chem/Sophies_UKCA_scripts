import cf
import numpy as np

dir_path = '/scratch/st838/netscratch/ukca_npy'
in_path = f'{dir_path}/cy731a.pl20150115.pp'

day = cf.read(in_path)
smallest = 1
item = ''
a64 = np.empty(0, dtype=np.float64)
a32 = np.empty(0, dtype=np.float32)
a16 = np.empty(0, dtype=np.float16)
for field in day:
  arr = field.array
  try:
    small = np.min(arr[np.nonzero(arr)])
  except:
    continue
  a64 = np.append(arr.astype(np.float64), a64)
  a32 = np.append(arr.astype(np.float32), a32)
  a16 = np.append(arr.astype(np.float16), a16)
  if small < smallest:
    smallest = small
    item = field.long_name
    print(f'{item} is {small}')
    
print(f'The smallest value in the dataset is {np.min(a64)}.')    # If +ve, use unsigned.
print(f'The smallest +ve value in the dataset is {item} = {smallest}.')
print(f'The smallest +ve value in the 64 bit array is {np.min(a64[np.nonzero(a64)])}')
print(f'The size of the 64 bit array is {(a64.itemsize * a64.size)/1000000000} gigabytes.')
print(f'The smallest +ve value in the 32 bit array is {np.min(a32[np.nonzero(a32)])}')
print(f'The size of the 32 bit array is {(a32.itemsize * a32.size)/1000000000} gigabytes.')
print(f'The smallest +ve value in the 16 bit array is {np.min(a16[np.nonzero(a16)])}')
print(f'The size of the 16 bit array is {(a16.itemsize * a16.size)/1000000000} gigabytes.')

if np.all(a32 == a64):
  word1 = 'No'
else:
  word1 = 'Some'
if np.all(a16 == a64):
  word2 = 'No'
else:
  word2 = 'Some'  

print(f'{word1} data were lost when reducing precision to 32 bit. {word2} data were lost when reducing precision to 16 bit.')
