'''
Name: Sophie Turner.
Date: 13/5/2025.
Contact: st838@cam.ac.uk.
Compare performace of random forests with and without corrected altitude and SZA.
Not for use with my standard .npy data before May 2025.
'''

import time
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.ensemble import RandomForestRegressor


def scale_sza_with_height(cos_theta0, altitude_km, pressure_pa):
    """
    Adjusts the cosine of the solar zenith angle (SZA) for a given altitude.
    Parameters:
    cos_theta0 (float): Cosine of the sea-level SZA (unitless, cos of radians)
    altitude_km (float): Observer's altitude above sea level (km)
    pressure_pa (float): Local air pressure (Pa)
    Returns:
    float: Adjusted cosine of the solar zenith angle at altitude (unitless, cos of radians)
    """
    # Earth's mean radius in km
    R = 6371.0 
    # Compute the geometric correction for altitude
    cos_theta_h = cos_theta0 + (altitude_km / (R + altitude_km)) * (1 - cos_theta0)
    # Compute atmospheric refraction correction (only applies for lower altitudes)
    theta_h = np.arccos(cos_theta_h)  # Convert back to radians
    refraction_angle_deg = (pressure_pa / 101325) * 0.0167 * np.tan(theta_h)  # Refraction in degrees
    refraction_angle_rad = np.radians(refraction_angle_deg)  # Convert to radians
    # Apply the refraction correction
    cos_theta_h = np.cos(theta_h - refraction_angle_rad)
    return(np.float32(cos_theta_h)) 
    

#print('\nWithout the correct altitudes.')
#data_file = f'{paths.npy}/test_day.npy'

print('\nWith the correct altitudes.')
data_file = f'{paths.npy}/test_day_corrected_alt.npy'

data = np.load(data_file)

# Remove upper stratosphere.
data = data[:, data[9] > 20]
# Remove night.
data = data[:, data[12] > 0]

print(data[con.alt, 0])
# Make sure altitude is in km.
#data[con.alt] = data[con.alt] * 85 # If using non-corrected alt.
data[con.alt] = data[con.alt] / 1000 # If using correct alt.
print(data[con.alt, 0])

# Test on lowest 10 km of altitude.
#data = data[:, data[con.alt] <= 10]

# Adjust the SZA if using corrected alt.
szas = data[10]
alts = data[con.alt]
press = data[9]
szas = scale_sza_with_height(szas, alts, press)
data[10] = szas

input_i = [0,1,2,3,4,5,8,9,10,11,12,16] # Original.
#target_i = np.arange(18, 88) # All.
target_i = 71 # H2O.
#target_i = 69 # NO3.
#target_i = 19 # NO2.
#target_i = 18 # O3.

# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, input_i, target_i)

# 90/10 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets)

# Make the regression model.
model = RandomForestRegressor(n_estimators=20, n_jobs=20, max_features=0.3, max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)
model.fit(in_train, out_train)
out_pred, maxe, mse, mape, smape, r2 = fns.test(model, in_test, out_test)

# View performance.
fns.show(out_test, out_pred, maxe, mse, mape, smape, r2)
