import numpy as np

# .pp is 16 GB.

# 83 items x 24 time x 85 height x 144 lat x 192 lon.
same_size = np.full((83, 24, 85, 144, 192), 0.0034657)
np.save(f'/scratch/st838/netscratch/tests/same_size.npy', same_size) # 35 GB.

elements = 83 * 24 * 85 * 144 * 192
flat_size = np.full(elements, 0.003432)
np.save(f'/scratch/st838/netscratch/tests/flat_size.npy', flat_size) # 35 GB.
