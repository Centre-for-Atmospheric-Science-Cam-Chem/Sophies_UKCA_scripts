# Update the a UKCA Fast-J .dat file based on GEOS-chem data for specific reactions.

import sys
from datetime import date

  
def find_lines2(data, name, lines, chars):
  # Choose lines to copy to UKCA, removing comments. 
  start = data.find(name)
  rest = data[start:]
  # Use min length line to ensure comments are not copied.
  keep = rest[:chars-1]
  # Copy each line.
  for _ in range(lines-1):
    next = rest.find('\n') + 1 
    rest = rest[next:]
    keep += '\n' + rest[:chars-1]
  return(keep)


def find_lines(data, name, lines, chars):
  # Choose lines to copy to UKCA, removing comments. 
  start = data.find(name)

  # Check that the name can be found.
  if start == -1:
    sys.exit(f'{name} could not be found in the GEOs-chem data file.')
  else:   
    found = True
    
  keep = ''  
  # Get all the data for the rxn.
  while(found):
    rest = data[start:]
    # Use min length line to ensure comments are not copied.
    keep += rest[:chars-1]
    # Copy each line.
    for _ in range(lines-1):
      next = rest.find('\n') + 1 
      rest = rest[next:]
      keep += '\n' + rest[:chars-1]
  
    # See if there is another wavelength for this rxn.
    next = rest.find('\n') + 1
    rest = rest[next:]
    rxn = rest[:7]
    found = name in rxn
    
    # Move to the next wavelength.
    start += data[start+1:].find(name)+1
   
  print(keep)
  exit()
  return(keep)
  

# Today's month and year. 
today = date.today()

# Files' directories.
dir_in = '/scratch/st838/netscratch/jHCHO_update/dat_files'
dir_out = '/scratch/st838/netscratch/tests'
# Input file paths.
geos_file = f'{dir_in}/GEOS_FJX.dat'
ukca_file_old = f'{dir_in}/FJX_spec_11May2020.dat'
# Output file path.
ukca_file_new = f'{dir_out}/FJX_spec_{today.strftime("%B%Y")}.dat'

# Corresponding names in UKCA and GEOS-chem
# of reactions which are to be updated in UKCA.
# [UKCA name, GEOS-chem name]
names = [['jhchoa', 'H2COa'],
         ['jhchob', 'H2COb']]

# Open but don't alter inputs.
geos_dat = open(geos_file, 'r')
ukca_dat = open(ukca_file_old, 'r')

# There are 3 lines of data for each rxn wavelength.
lines = 3
# Minimum characters in each line.
chars = len(ukca_dat.readlines()[2])

# Read the file.
geos_data = geos_dat.read()

for rxn in names:
  geos_name = rxn[1]
  # Find the data for this reaction.
  rxn_data = find_lines(geos_data, geos_name, lines, chars)
    

# Output.
out_dat = open(ukca_file_new, 'w')
