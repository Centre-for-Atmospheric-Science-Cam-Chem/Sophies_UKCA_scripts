import cf

# Read in the .pp file and select a field.
test_file = '/scratch/st838/netscratch/ukca_npy/cy731a.pl20151015.pp'
field = cf.read(test_file, select='stash_code=50500')[0]

for i in range(85):
  print('model level number =', i+1)
  z = field.coord('atmosphere_hybrid_height_coordinate').data[i]
  z = z * 85
  print('altitude =', z, 'km')
  print()
