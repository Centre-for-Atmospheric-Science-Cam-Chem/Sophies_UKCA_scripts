'''
Name: Sophie Turner.
Date: 30/4/2025.
Contact: st838@cam.ac.uk.
Perform all useful data conversions from UM format to format useful for ML. Conversions are:
Altitude from a proportion of the top level to kilometres,
Altitude from levels based at sea-level to altitude with orography,
Longitude from +=180 degrees to 360 degree scale,
Solar zenith angle from cosine of radians to degrees,
Solar zenith angle from surface-level to all levels scaled by height and refraction,
New data for summed cloud in columns above each grid box.
'''

import math
import numpy as np
import constants as con
import functions as fns
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


data_file = f'{paths.npy}/20160701_midday.npy'
data = np.load(data_file)

# Longitude from 360 degree scale to +-180 degrees.
lons = data[con.lon]
lons[lons > 180] -= 360
data[con.lon] = lons

# Altitude from levels based at sea-level to altitude with orography.
alts = data[con.alt]
orog = data[con.orog]
# See if there's a way to apply rob's fn to each gridbox/whole grid.


# Altitude from a proportion of the top level to kilometres.
alts = alts * 85

# Solar zenith angle from surface-level to all levels scaled by height and refraction.
szas = data[con.sza] 
press = data[con.pressure]
szas = scale_sza_with_height(szas, alts, press)

# Solar zenith angle from cosine of radians to degrees.
szas = szas * 180 / math.pi
data[con.sza] = szas

# New data for summed cloud in columns above each grid box.
data = fns.sum_cloud(data)

# Save changes.
#np.save(data_file, data)
