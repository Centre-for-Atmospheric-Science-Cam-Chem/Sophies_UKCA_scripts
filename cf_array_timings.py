import cf
import time

# Read in the .pp file and select a 4d field.
test_file = '/scratch/st838/cy731a.pl20151015.pp'
field = cf.read(test_file, select='stash_code=50500')[0]
# 24 time x 85 height x 144 lat x 192 lon.

# Time how long it takes to fetch the numerical data and make a np array.
start = time.time()
field_np = field.array
end = time.time()
elapsed = end - start

print(f'The {type(field_np)} of shape {field_np.shape} took {round(elapsed, 1)} seconds.')
