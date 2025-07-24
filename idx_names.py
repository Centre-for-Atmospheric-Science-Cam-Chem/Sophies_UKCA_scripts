'''
Name: Sophie Turner.
Date: 18/2/2025.
Contact: st838@cam.ac.uk
Names and indices of fields for easier indexing and analysis of ML stuff. Import into scripts.
'''

import constants as con

# [index, 'UM stash code', 'full, formatted name', 'short, simple name']
# The short names are for file paths and stuff like that so they contain no whitespaces.
idx_names = [
[0, '', 'hour of day', 'hour'], 
[1, '', 'altitude', 'altitude'], # Level, converted to km later.
[2, '', 'latitude / deg N', 'latitude'], 
[3, '', 'longitude / deg E', 'longitude'],
[4, '', 'days since start of 1/1/2015', 'date-time'],
[5, 'm01s01i010', 'specific humidity', 'humidity'], 
[6, 'm01s00i266', 'cloud fraction', 'cloud'],
[7, 'm01s00i408', 'air pressure / Pa', 'pressure'], 
[8, 'm01s01i140', 'solar zenith angle', 'sza'], # Cosine of sza in rad, later converted to deg.
[9, 'm01s01i217', f'upward shortwave flux / {con.Wperm2}', 'up_sw_flux'],
[10, 'm01s01i218', f'downward shortwave flux / {con.Wperm2}', 'down_sw_flux'],
[11, 'm01s02i217', f'upward longwave flux / {con.Wperm2}', 'up_lw_flux'],
[12, 'm01s02i218', f'downward longwave flux / {con.Wperm2}', 'down_lw_flux'],
[13, 'm01s16i004', 'temperature / K', 'temperature'],
[14, 'm01s50i219', 'ozone column', 'O3_col'],
[15, 'm01s50i228', 'ozone', 'O3'], # O3 to O1D. Same as 78 but not from Fast-J? Not in ASAD.
[16, 'm01s50i229', f'NO{con.sub2}', 'NO2'], # Also not in ASAD.
[17, 'm01s50i500', 'organic nitrate J rates (summed)', 'nitrates'],
[18, 'm01s50i501', 'formaldehyde (radical)', 'HCHOr'],
[19, 'm01s50i502', 'formaldehyde (molecular)', 'HCHOm'],
[20, 'm01s50i503', f'MeCOCHO', 'MeCOCHO'],
[21, 'm01s50i504', 'CO production (summed)', 'prodCO'],
[22, 'm01s50i505', 'stratospheric OH-production (summed)', 'prodOH'],
[23, 'm01s50i506', f'O{con.sub2}', 'O2'],
[24, 'm01s50i507', f'Cl{con.sub2}O{con.sub2}', 'Cl2O2'], # Stratospheric?
[25, 'm01s50i508', f'stratospheric NO{con.sub3}', 'NO3strat'],
[26, 'm01s50i509', f'summed O({con.sup1}D)', 'O1D'], 
[27, 'm01s50i510', f'RU{con.sub1}{con.sub2}NO{con.sub3}', 'RU12NO3'],
[28, 'm01s50i511', 'OCS', 'OCS'],
[29, 'm01s50i512', f'H{con.sub2}SO{con.sub4}', 'H2SO4'],
[30, 'm01s50i513', f'SO{con.sub3}', 'SO3'],
[31, 'm01s50i514', f'MeONO{con.sub2}', 'MeONO2'],
[32, 'm01s50i515', 'NALD', 'NALD'],
[33, 'm01s50i516', 'isoprene nitrate', 'ISON'],
[34, 'm01s50i517', f'EtONO{con.sub2}', 'EtONO2'],
[35, 'm01s50i518', f'RN{con.sub1}{con.sub0}NO{con.sub3}', 'RN10NO3'],
[36, 'm01s50i519', f'i-PrONO{con.sub2}', 'i-PrONO2'],
[37, 'm01s50i520', f'RN{con.sub1}{con.sub3}NO{con.sub3} {con.to} MeCHO', 'RN13NO3-MeCHO'],
[38, 'm01s50i521', f'RN{con.sub1}{con.sub3}NO{con.sub3} {con.to} CARB{con.sub1}{con.sub1}A', 'RN13NO3-CARB11A'],
[39, 'm01s50i522', f'RN{con.sub1}{con.sub6}NO{con.sub3}', 'RN16NO3'],
[40, 'm01s50i523', f'RN{con.sub1}{con.sub9}NO{con.sub3}', 'RN19NO3'],
[41, 'm01s50i524', f'RA{con.sub1}{con.sub3}NO{con.sub3}', 'RA13NO3'],
[42, 'm01s50i525', f'RA{con.sub1}{con.sub6}NO{con.sub3}', 'RA16NO3'],
[43, 'm01s50i526', f'RA{con.sub1}{con.sub9}NO{con.sub3}', 'RA19NO3'],
[44, 'm01s50i527', f'RTX{con.sub2}{con.sub4}NO{con.sub3}', 'RTX24NO3'],
[45, 'm01s50i528', f'RU{con.sub1}{con.sub0}NO{con.sub3}', 'RU10NO3'],
[46, 'm01s50i529', f'RU{con.sub1}{con.sub4}NO{con.sub3}', 'RU14NO3'],
[47, 'm01s50i530', f'CARB{con.sub3} {con.to} HO{con.sub2}', 'CARB3-HO2'],
[48, 'm01s50i531', f'CARB{con.sub6}', 'CARB6'],
[49, 'm01s50i532', f'CARB{con.sub3} {con.to} H{con.sub2}', 'CARB3-H2'],
[50, 'm01s50i533', f'CARB{con.sub3} {con.to} HCHO', 'CARB3-HCHO'],
[51, 'm01s50i540', f'MeCHO {con.to} MeOO', 'MeCHO-MeOO'],
[52, 'm01s50i541', 'propanal', 'propanal'],
[53, 'm01s50i542', f'HOCH{con.sub2}CHO', 'HOCH2CHO'],
[54, 'm01s50i543', f'UCARB{con.sub1}{con.sub2} {con.to} MeCO{con.sub3}', 'UCARB12-MeCO3'],
[55, 'm01s50i544', f'UCARB{con.sub1}{con.sub2} {con.to} CARB{con.sub7}', 'UCARB12-CARB7'],
[56, 'm01s50i545', f'NUCARB{con.sub1}{con.sub2} {con.to} HUCARB{con.sub9}', 'NUCARB12-HUCARB9'],
[57, 'm01s50i546', f'TNCARB{con.sub1}{con.sub0}', 'TNCARB10'],
[58, 'm01s50i547', f'RTN{con.sub1}{con.sub0}OOH', 'RTN10OOH'],
[59, 'm01s50i548', f'DHPCARB{con.sub9}', 'DHPCARB9'], # Identical to 67.
[60, 'm01s50i549', f'HPUCARB{con.sub1}{con.sub2} {con.to} HUCARB{con.sub9}', 'HPUCARB12-HUCARB9'],
[61, 'm01s50i550', f'HPUCARB{con.sub1}{con.sub2} {con.to} CARB{con.sub7}', 'HPUCARB12-CARB7'],
[62, 'm01s50i551', f'HUCARB{con.sub9}', 'HUCARB9'],
[63, 'm01s50i552', f'DHPR{con.sub1}{con.sub2}OOH', 'DHPR12OOH'],
[64, 'm01s50i553', f'DHCARB{con.sub9}', 'DHCARB9'],
[65, 'm01s50i554', f'NUCARB{con.sub1}{con.sub2} {con.to} CARB{con.sub7}', 'NUCARB12-CARB7'],
[66, 'm01s50i555', f'NO{con.sub3}', 'NO3'], 
[67, 'm01s50i556', f'DHPCARB{con.sub9}', 'DHPCARB9'], # Identical to 59.
[68, 'm01s50i557', f'H{con.sub2}O', 'H2O'],
[69, 'm01s50i558', 'methane', 'CH4'],
[70, 'm01s50i559', 'HOBr', 'HOBr'],
[71, 'm01s50i560', 'HOCl', 'HOCl'],
[72, 'm01s50i561', f'HNO{con.sub3}', 'HNO3'],
[73, 'm01s50i562', f'HNO{con.sub4}', 'HNO4'],
[74, 'm01s50i563', f'H{con.sub2}O{con.sub2}', 'H2O2'],
[75, 'm01s50i564', 'MeOOH', 'MeOOH'],
[76, 'm01s50i565', f'O{con.sub2} {con.to} O({con.sup3}P)', 'O2-O3P'],
[77, 'm01s50i566', f'O{con.sub2} {con.to} O({con.sup1}D)', 'O2-O1D'],
[78, 'm01s50i567', 'ozone', 'O3'], # Same as 15 but this one is in ASAD.
[79, 'm01s50i568', f'N{con.sub2}O', 'N2O'],
[80, 'm01s50i569', 'methacrolein', 'MACR'],
[81, 'm01s50i570', 'MACROOH', 'MACROOH'],
[82, 'm01s50i571', f'MeCHO {con.to} CH{con.sub4}', 'MeCHO-CH4'],
[83, 'm01s50i615', f'NRU{con.sub1}{con.sub2}OOH {con.to} CO', 'NRU12OOH-CO'],
[84, 'm01s50i616', f'NRU{con.sub1}{con.sub2}OOH {con.to} CARB{con.sub3}', 'NRU12OOH-CARB3']
]

# Items that have data present in the tropospheric npy datasets.
trop_idxs = list(range(27)) + [28,30,31,32,33,51,52,66,68] + list(range(70,77)) + list(range(78,83))
idx_names_trop = [idx_names[i] for i in trop_idxs]

# Items that have data present in the strat-trop npy datasets.
strat_trop_idxs = trop_idxs + [29,69,77]
strat_trop_idxs.sort()
idx_names_strat_trop = [idx_names[i] for i in strat_trop_idxs]
