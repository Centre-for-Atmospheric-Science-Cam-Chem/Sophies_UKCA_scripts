'''
Name: Sophie Turner.
Date: 13/5/2025.
Contact: st838@cam.ac.uk.
Calculate altitude for every grid box.
This is likely to take 290 hours for one timestep of the whole grid
or 7000 hours for a full day.
'''

import time
import numpy as np
import constants as con
import functions as fns
import file_paths as paths

rho_levels = dict(
  z_top_of_model=85000.00,
  first_constant_r_rho_level=51, 
  eta_rho=[
    0.1176471E-03,   0.4313726E-03,   0.9019608E-03,   0.1529412E-02,   0.2313725E-02,
    0.3254902E-02,   0.4352941E-02,   0.5607843E-02,   0.7019607E-02,   0.8588235E-02,
    0.1031373E-01,   0.1219608E-01,   0.1423529E-01,   0.1643137E-01,   0.1878431E-01,
    0.2129412E-01,   0.2396078E-01,   0.2678431E-01,   0.2976470E-01,   0.3290196E-01,
    0.3619608E-01,   0.3964706E-01,   0.4325490E-01,   0.4701960E-01,   0.5094118E-01,
    0.5501961E-01,   0.5925490E-01,   0.6364705E-01,   0.6819607E-01,   0.7290196E-01,
    0.7776470E-01,   0.8278431E-01,   0.8796078E-01,   0.9329412E-01,   0.9878433E-01,
    0.1044314E+00,   0.1102354E+00,   0.1161964E+00,   0.1223144E+00,   0.1285897E+00,
    0.1350224E+00,   0.1416128E+00,   0.1483616E+00,   0.1552695E+00,   0.1623374E+00,
    0.1695668E+00,   0.1769595E+00,   0.1845180E+00,   0.1922454E+00,   0.2001459E+00,
    0.2082247E+00,   0.2164882E+00,   0.2249446E+00,   0.2336039E+00,   0.2424783E+00,
    0.2515826E+00,   0.2609347E+00,   0.2705562E+00,   0.2804726E+00,   0.2907141E+00,
    0.3013166E+00,   0.3123218E+00,   0.3237787E+00,   0.3357441E+00,   0.3482838E+00,
    0.3614739E+00,   0.3754014E+00,   0.3901665E+00,   0.4058831E+00,   0.4226810E+00,
    0.4407075E+00,   0.4601292E+00,   0.4811340E+00,   0.5039334E+00,   0.5287649E+00,
    0.5558944E+00,   0.5856187E+00,   0.6182693E+00,   0.6542144E+00,   0.6938630E+00,
    0.7376686E+00,   0.7861323E+00,   0.8398075E+00,   0.8993046E+00,   0.9652942E+00]
)    

eta_first_constant_rho = rho_levels['eta_rho'][rho_levels['first_constant_r_rho_level'] - 1] # This can be a global constant.

print('\nWith the correct altitudes.')
data_file = f'{paths.npy}/test_day.npy'
data = np.load(data_file) 

# Altitudes.
level = data[1].copy()
ground = data[7]

start = time.time()

# Calculate alt for every box without using the function.
for point_i in range(len(data[0])):
  # Get the unique list of proportinal altitude levels.
  alt_levels = np.unique(level)
  # Get the proportional altitude level.
  level_point = level[point_i]
  # Find its index in the list of all proportional altitude levels.
  alt_i = np.where(alt_levels == level_point)[0][0]
  # Find the eta_rho value at that index.
  eta = rho_levels['eta_rho'][alt_i]
  # Get the ground height.
  ground_point = ground[point_i] 
  # Apply the claculation.
  eta = rho_levels['eta_rho'][alt_i]
  if alt_i >= rho_levels['first_constant_r_rho_level'] :
    # Not terrain following.
    sigma = 0
  else:
    # Scaling factor for topography following coords.
    sigma = (1.0 - (eta / eta_first_constant_rho))**2
  delta = eta * 85000
  calc_alt = delta + (sigma * ground_point)
  # Replace altitude with the correct value.
  data[1, point_i] = calc_alt 
  if point_i == 1000:
    end = time.time()
    seconds = end - start
    print(f'The first 1000 loops of {len(data[0])} data points took {seconds} seconds.')   
  
end = time.time()
seconds = end - start
print(f'The loop of {len(data[0])} data points took {seconds} seconds.')  

# Check that they are different.
plt.scatter(level, data[1])
plt.xlabel('Altitude before orography calculation')
plt.ylabel('Altitude after orography calculation')
plt.show()

