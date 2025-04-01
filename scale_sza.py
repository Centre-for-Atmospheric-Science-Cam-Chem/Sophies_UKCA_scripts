'''
Name: Sophie Turner.
Date: 28/3/2025.
Contact: st838@cam.ac.uk.
Test scaling solar zenith angle with vertical height, to match Fast-J.
Generates a new training dataset to use to train a forest.
Then you can compare that forest to the original unscaled ones.
'''

import numpy as np
import constants as con
import file_paths as paths


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


'''
# TEST: use a test full-res column.
data = np.load('/scratch/st838/netscratch/data/ukca_npys/20150320.npy')
lat = 53.125
lon = 0.9375 
data = data[:, (data[con.lat] == lat) & (data[con.lon] == lon) & (data[con.hour] == 12)]
'''

# Load training dataset.
print('\nLoading data.')
filename = 'low_res_yr_500k'
data = np.load(f'{paths.npy}/{filename}.npy') 
print(data.shape) # (85, 182000000)

# Scale SZAs.
print('Scaling solaz zenith angles.') 
szas = data[con.sza]
alts = data[con.alt]  

# Try to replicate the way Fast-J scales SZA to get similar retsuls.
metres = alts * 85000
# Earth's radius (m).
rad = 6.371e6
szas_geo = np.sqrt(1 - ((rad / (rad + metres))**2) * (1 - szas**2))

# Try scaling based on another suggested method.
press = data[con.pressure]
kms = alts * 85
szas_refrac = scale_sza_with_height(szas, kms, press)

# Save the new training dataset.
data[con.sza] = szas_refrac 
out_path = f'{paths.npy}/{filename}_scaled_sza_refrac.npy'
print(f'Saving new dataset at {out_path}.')
np.save(out_path, data)
print('Done.')
