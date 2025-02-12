'''
Name: Sophie Turner.
Date: 24/9/2024.
Contact: st838@cam.ac.uk
Constants and variables which are frequently used by ML scripts.
Files are located at scratch/st838/netscratch.
'''

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Annoying subscripts & superscripts for displaying units etc. on plots.
sub2 = '\u2082'
sub3 = '\u2083'
sub4 = '\u2084'
sup1 = '\u00b9'
sup2 = '\u00b2'
supminus = '\u207b'
r2 = f'R{sup2}'
pers = f's{supminus}{sup1}' 
Wperm2 = f'Wm{supminus}{sup2}'
rxnO3 = f'O{sub3} -> O{sub2} + O({sup1}D)'

# Indices of some common features to use as inputs and outputs.
# Physics.
# Number of fields in 2D dataset.
n_phys = 15
phys_all = np.arange(n_phys, dtype=np.int16)
input_names = ['hour of day', 'altitude / km', 'latitude / deg', 'longitude / deg', 'days since 1/1/2015',
               'specific humidity', 'cloud fraction', 'pressure / Pa', 'solar zenith angle / degrees', 
              f'upward shortwave flux / {Wperm2}', f'downward shortwave flux / {Wperm2}', 
              f'upward longwave flux / {Wperm2}', f'downward longwave flux / {Wperm2}', 'temperature / K',
	      'ozone column']
input_names_short = ['hour', 'alt', 'lat', 'lon', 'days', 'humidity', 'cloud', 'pressure', 
                     'sza', 'up_sw_flux', 'down_sw_flux', 'up_lw_flux', 'down_lw_flux', 'temp', 'O3_col']
# Indices of individual physics fields. 
hour = 0
alt = 1
lat = 2
lon = 3
days = 4
humidity = 5
cloud = 6
pressure = 7
sza = 8
up_sw_flux = 9
down_sw_flux = 10
up_lw_flux = 11
down_lw_flux = 12
temp = 13
o3_col = 14
# Without pointless fields.
phys_main = [hour, alt, lat, lon, days, humidity, cloud, pressure, sza, up_sw_flux, down_sw_flux, temp]
# Physics inputs chosen by feature selection.
phys_best = [alt, pressure, sza, up_sw_flux, down_sw_flux]

# J rates.
# Number of fields in 2D dataset.
n_J = 70
n_fields = n_phys + n_J
J_all = np.arange(n_phys, n_fields, dtype=np.int16)
NO2 = 16
HCHOr = 18 # Radical product.
HCHOm = 19 # Molecular product.
NO3 = 66
HOCl = 71
H2O2 = 74
O3 = 78 # O(1D) product.
# J rates which are not summed or duplicate fg. with usually zero rates removed.
#           0  1 2  3 4  5  6 7  8 9 10 11 1213 14 1516 17 1819 20 21   
J_core    = [16,18,19,20,24,28,30,31,33,51,52,66,70,71,72,73,74,75,78,79,80,82]
J_core_cc = [17,19,20,21,25,29,31,32,34,52,53,67,71,72,73,74,75,76,79,80,81,83] # With cloud col.
J_names = [f'NO{sub2}', 'formaldehyde (radical)', 'formaldehyde (molecular)', f'CH{sub3}COCHO',
           f'Cl{sub2}O{sub2}', 'OCS', f'SO{sub3}', f'MeONO{sub2}', 'ISON', 'MeCHO -> MeOO',
	   'propanal', f'NO{sub3}', 'HOBr', 'HOCl', f'HONO{sub2}', f'HO{sub2}NO{sub2}',
	   f'H{sub2}O{sub2}', 'MeOOH', f'O{sub3}', f'N{sub2}O', 'methacrolein', f'MeCHO -> CH{sub4}']
J_names_short = ['NO2', 'HCHOr', 'HCHOm', 'CH3COCHO', 'Cl2O2', 'OCS', 'SO3', 'MeONO2', 'ISON',
                 'MeCHO MeOO', 'propanal', 'NO3', 'HOBr', 'HOCl', 'HONO2', 'HO2NO2', 'H2O2',
		 'MeOOH', 'O3', 'N2O', 'MACR', 'MeCHO CH4']
# Aldehydes in J_core.
alds = [18,19,20,51,52,80,82]
ald_names = ['formaldehyde (radical)', 'formaldehyde (molecular)', f'CH{sub3}COCHO', 
             'MeCHO -> MeOO', 'propanal', 'methacrolein', f'MeCHO -> CH{sub4}']
# Nitrates.
nitrates = [31,33]
# Nitric acids.
nit_acids = [72,73]
# Weak halogen acids.
hal_acids = [70,71]
# Peroxides.
perox = [24,74]

# Locations for columns (hour, lat, lon).
np12 = [12, 89.375, 0.9375, 'North pole at midday', 'np12']
cam12 = [12, 51.875, 0.9375, 'Cambridge at midday', 'cam12']
gg12 = [12, 0.625, 0.9375, 'Gulf of Guinea at midday', 'gg12']
ac12 = [12, -23.125, 0.9375, 'Atlantic Capricorn at midday', 'ac12']
sp12 = [12, -89.375, 0.9375, 'South pole at midday', 'sp12']

# Gigabyte, for memory tests. 
GB = 1000000000
# For consistent pseudo-random states.
seed = 6
# For consistent numpy random numbers.
rng = np.random.default_rng(seed)

# Colourmap used for % differences with green for zero.
cmap_diff = LinearSegmentedColormap.from_list("Cdiff", ["darkblue", "blue", "deepskyblue", "cyan", "lawngreen", "yellow", "orange", "red", "firebrick"]) 
# Colourmap used for R2 scores (ideally from 0.5 to 1).
cmap_r2 = LinearSegmentedColormap.from_list("Cr2", ["red", "orange", "yellow", "lime"])
