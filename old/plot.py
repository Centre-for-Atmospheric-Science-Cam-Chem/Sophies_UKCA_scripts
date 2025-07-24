'''
Plot ML outputs from npy files.
'''

import numpy as np
import file_paths as paths
import prediction_fns_shared as fns

pred_file = f'{paths.npy}/out_pred.npy'
test_file = f'{paths.npy}/out_test.npy'
out_pred = np.load(pred_file)
out_test = np.load(test_file)
fns.show(out_test, out_pred, 0, 0)
