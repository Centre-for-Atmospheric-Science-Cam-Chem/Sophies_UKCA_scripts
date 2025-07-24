import cf
field = cf.read('/scratch/st838/netscratch/ukca_npy/cy731a.pl20151015.pp', select='stash_code=50500')[0]
alts = field.coord('atmosphere_hybrid_height_coordinate').array
print(alts)
print()
alts = alts * 85
for alt in alts:
  print(round(alt, 2))
