# Test cloud summing fn.

import numpy as np
import file_paths as paths
import functions as fns

data_file = f'{paths.npy}/20150115.npy'
out_file = f'{paths.npy}/20150115cc.npy'
data = np.load(data_file)
fns.sum_cloud(data, out_file)
