'''
Name: Sophie Turner.
Date: 4/4/2025.
Contact: st838@cam.ac.uk.
Implement Rob's level-to-altitude conversion on the whole grid.
It might seem like there's a lot of unnecessary code in the loop, 
but that's code I'll need later to overwrite altitude values in the datasets.
'''

import numpy as np
import constants as con
import file_paths as paths
import matplotlib.pyplot as plt 


# Rob's code.
def lvl_to_alt(grid_dict: dict, orography: float, 
                             include_surface: bool = False) -> list:
    """
    Function to calculate model level heights above ground for a single vertical column. 
    Args:
        grid_dict (dict): dictionary version of the data held in the level control file (see below for 85-level grid). 
        orography (float): ground height for a vertical column, usually from STASH section 0, item 33.
        include_surface (bool, optional): whether or not to include the lowest model level. Defaults to False.
    Returns:
        list: list of altitudes of model levels. 
    """
    # checking sigma
    eta_first_constant_rho = grid_dict['eta_rho'][
        grid_dict['first_constant_r_rho_level'] - 1]
    z_list = []
    if include_surface:
        starting_index = 0
    else:
        starting_index = 1
    for i in range(starting_index, len(grid_dict['eta_theta'])): # Previously eta_theta.
        eta = grid_dict['eta_theta'][i] # # Previously eta_theta.
        if i >= grid_dict['first_constant_r_rho_level'] :
            # i.e. not terrain following
            sigma = 0
        else:
            # scaling factor for topography following coords
            sigma = (1.0 - (eta / eta_first_constant_rho))**2
        # with no terrain use eta.
        delta = eta * grid_dict['z_top_of_model']
        z_list.append(delta + (sigma * orography))
    return z_list
    
# These arrays are proportional heights out of 85km of each of the 85 levels.    
levels = dict(
  z_top_of_model=85000.00,
  first_constant_r_rho_level=51, 
  eta_theta=[
    0.0000000E+00,   0.2352941E-03,   0.6274510E-03,   0.1176471E-02,   0.1882353E-02,
    0.2745098E-02,   0.3764706E-02,   0.4941176E-02,   0.6274510E-02,   0.7764705E-02,
    0.9411764E-02,   0.1121569E-01,   0.1317647E-01,   0.1529412E-01,   0.1756863E-01,
    0.2000000E-01,   0.2258823E-01,   0.2533333E-01,   0.2823529E-01,   0.3129411E-01,
    0.3450980E-01,   0.3788235E-01,   0.4141176E-01,   0.4509804E-01,   0.4894118E-01,
    0.5294117E-01,   0.5709804E-01,   0.6141176E-01,   0.6588235E-01,   0.7050980E-01,
    0.7529411E-01,   0.8023529E-01,   0.8533333E-01,   0.9058823E-01,   0.9600001E-01,
    0.1015687E+00,   0.1072942E+00,   0.1131767E+00,   0.1192161E+00,   0.1254127E+00,
    0.1317666E+00,   0.1382781E+00,   0.1449476E+00,   0.1517757E+00,   0.1587633E+00,
    0.1659115E+00,   0.1732221E+00,   0.1806969E+00,   0.1883390E+00,   0.1961518E+00,
    0.2041400E+00,   0.2123093E+00,   0.2206671E+00,   0.2292222E+00,   0.2379856E+00,
    0.2469709E+00,   0.2561942E+00,   0.2656752E+00,   0.2754372E+00,   0.2855080E+00,
    0.2959203E+00,   0.3067128E+00,   0.3179307E+00,   0.3296266E+00,   0.3418615E+00,
    0.3547061E+00,   0.3682416E+00,   0.3825613E+00,   0.3977717E+00,   0.4139944E+00,
    0.4313675E+00,   0.4500474E+00,   0.4702109E+00,   0.4920571E+00,   0.5158098E+00,
    0.5417201E+00,   0.5700686E+00,   0.6011688E+00,   0.6353697E+00,   0.6730590E+00,
    0.7146671E+00,   0.7606701E+00,   0.8115944E+00,   0.8680208E+00,   0.9305884E+00,
    0.1000000E+01],
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
# End of Rob's code.

# A full day of hourly UM output data as a 2d numpy array.
np_file = f'{paths.npy}/test_day.npy'
print('\nLoading data.')
data = np.load(np_file) 
print(data.shape) # (85, 56401920) or (85, 2350080)
print('Calculating model level altitudes.')

# Select one timestep (midday).
data = data[:, (data[con.hour] == 12)]

# Get latitudes and longitudes in the data.
lats = np.unique(data[con.lat]) 
lons = np.unique(data[con.lon]) 
# Select one latitude.
data = data[:, (data[con.lat] == lats[71])] # Middle = lats[71].

# Fetch the current values for altitude. They are the same across the entire lat-lon grid.
alts_sea = data[con.alt].copy()
alts_sea_sequence = np.unique(alts_sea)

# Start the ground height at sea level and keep track of the ground surface.
ground = 0
surface = []

# Get the vertical columns along one latitude in the dataset.
# For each lon...
for lon in lons: 
  # Get the data in that column.
  col_idx = np.where(data[con.lon] == lon)[0]
  col = data[:, col_idx] 
  
  '''
  # Dummy data for ground height for each column, because I haven't yet got the real data for this.
  # Let it go up or down by <= 100 metres from the previous value so that it's not ridiculously jagged.
  ground += (con.rng.random(dtype=np.float32) - 0.5) * 100
  # Don't let it go below sea level for now. 
  if ground < 0:
    ground = 0      
  '''
  
  # Ground height for each column.
  ground = col[7][0]

  # Add this bit of ground to the surface.
  surface.append(ground) 
  
  # Get the altitudes for each level in the column.
  alts_hills = lvl_to_alt(levels, ground, False)

  # Update the column's altitudes in the dataset.
  # To do: Add a new column of data for altitude instead of overwriting alt.
  data[con.alt, col_idx] = alts_hills  

# See how altitude compares to model level along this mid-latitude sample slice. 
# Hidden lines just to get the legend to work.
plt.plot(lons, surface, color='limegreen', label='Model levels without orography')
plt.plot(lons, surface, color='mediumorchid', label='Model levels with orography')
# A hoizontal line for each model level. Just look at the lowest 20 levels.
for lvl in range(85): 
  alt_idx = np.where(alts_sea == alts_sea_sequence[lvl])[0]
  # The actual lines for levels with and without orography.
  plt.plot(lons, alts_sea[alt_idx] * 85000, color='limegreen')
  plt.plot(lons, data[con.alt, alt_idx], color='mediumorchid', linestyle='--')
# The actual surface line.
plt.plot(lons, surface, color='orange', label='Ground surface') # label='Ground surface (random dummy data)'
plt.title('A test of level altitude calculation along one latitude')
plt.xlabel('Longitude')
plt.ylabel('Altitude')
plt.legend()
plt.show()
plt.close()
