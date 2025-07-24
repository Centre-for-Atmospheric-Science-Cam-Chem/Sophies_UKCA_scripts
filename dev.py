import numpy as np
import file_paths as paths

dims = np.load(f'{paths.npy}/dims.npy')
print(dims.shape)
print(dims)
