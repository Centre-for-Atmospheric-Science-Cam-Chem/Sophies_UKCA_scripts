import numpy as np
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
  

def shrink(out_test, out_pred):
  # Don't plot an unnecessary number of data points i.e. >10000.  
  # Make them the right shape.
  out_test = out_test.squeeze()
  out_pred = out_pred.squeeze()
  length = len(out_pred)
  if length > 10000:
    # Plotting this many datapoints is excessive and costly. Reduce it to 10000.
    idxs = np.int16(np.linspace(0, length, 10000))
    out_test = out_test[idxs]
    out_pred = out_pred[idxs]
    del(idxs)
    # Choose opacity of points.
    alpha = 0.1
  elif length > 100:
    alpha = 0.5
  else:
    alpha = 1
  return(out_test, out_pred, alpha)


def force_axes():
  # Make plot axes exactly the same.
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
  

out_pred = f'{paths.results}/out_pred_forest2.npy'
out_test = f'{paths.results}/out_test_forest2.npy'

out_pred = np.load(out_pred)
out_test = np.load(out_test)

out_pred = out_pred.squeeze()
out_test = out_test.squeeze()

r2 = round(r2_score(out_test, out_pred), 3) 

# Don't plot >10000 points.
out_test, out_pred, a = shrink(out_test, out_pred)
plt.scatter(out_test, out_pred, alpha=a)
# Force axes to be identical.
force_axes()
plt.title(f'JNO2, low resolution year. R\u00b2={r2}')
plt.xlabel('J rate from Fast-J / s\u207b\u00b9')
plt.ylabel('J rate from random forest / s\u207b\u00b9')
plt.show() 
plt.close()

'''

data_path = f'{paths.data}/low_res_yr_2015.npy'
data = np.load(data_path)
alt = data[1, 0]
lat = 51.875
lon = data[3, 0]
data = data[:, np.where(data[1] == alt)].squeeze()
data = data[:, np.where(data[2] == lat)].squeeze()
data = data[:, np.where(data[3] == lon)].squeeze()
dt = data[4]
o3 = data[78]

plt.scatter(dt, o3)
plt.title('Ozone photolysis rate near Cambridge, 1-year low-resolution dataset.')
plt.xlabel('Date-time, days since 01/01/2015')
plt.ylabel('J O3')
plt.show() 
plt.close()

'''
