'''
Name: Sophie Turner.
Date: 26/10/2023.
Contact: st838@cam.ac.uk
Convert UM stash codes for J rates to names which match ATom data.
Used by scripts located at scratch/st838/netscratch 
for use on Cambridge chemistry department's atmospheric servers. 
'''

# ['UM stash code', 'ATom field name or new name']
code_names = [['specific_humidity'   , 'SPECIFIC HUMIDITY'],
              ['UM_m01s00i266_vn1300', 'CLOUD FRACTION'],
              ['air_pressure'        , 'PRESSURE'],
              ['UM_m01s01i142_vn1300', 'COS SOLAR ZENITH ANGLE'],
	      ['UM_m01s01i217_vn1300', 'UPWARD SW FLUX'],
              ['UM_m01s01i218_vn1300', 'DOWNWARD SW FLUX'],
              ['UM_m01s02i217_vn1300', 'UPWARD LW FLUX'],
              ['UM_m01s02i218_vn1300', 'DOWNWARD LW FLUX'],
	      ['air_temperature'     , 'TEMPERATURE'],
	      ['UM_m01s30i206_vn1300', 'RELATIVE HUMIDITY ON PLEV/UV GRID'],
              ['UM_m01s30i296_vn1300', 'RELATIVE HUMIDITY ON PLEV/T GRID'],
	      ['UM_m01s30i301_vn1300', 'HEAVYSIDE FN ON P LEV/UV GRID'], 
              ['UM_m01s30i304_vn1300', 'HEAVYSIDE FN ON P LEV/T GRID'], 
              ['UM_m01s50i228_vn1300', 'O1D'], 
	      ['UM_m01s50i229_vn1300', 'jNO2_NO_O3P'], # Matched to ATom
	      ['UM_m01s50i500_vn1300', 'organic nitrate J rates (summed)'],
              ['UM_m01s50i501_vn1300', 'jCH2O_H_HCO'], # HCHO radical matched to ATom
              ['UM_m01s50i502_vn1300', 'jCH2O_H2_CO'], # HCHO molecular matched to ATom
              ['UM_m01s50i503_vn1300', 'jCH3COCHO'], # MGLY (summed) matched to ATom.
              ['UM_m01s50i504_vn1300', 'Other CO producing rxns J rates (summed)'],
              ['UM_m01s50i505_vn1300', 'H and O stuff J rates (summed)'],
              ['UM_m01s50i506_vn1300', 'O2 J rates (summed)'],
              ['UM_m01s50i507_vn1300', 'jCl2O2'],
              ['UM_m01s50i508_vn1300', 'jNO3_NO_O2 b'], # Matched to ATom but stratospheric.
              ['UM_m01s50i509_vn1300', 'jO1D (summed)'],
              ['UM_m01s50i510_vn1300', 'jRU12NO3'],
              ['UM_m01s50i511_vn1300', 'jCOS'],
              ['UM_m01s50i512_vn1300', 'jH2SO4'],
              ['UM_m01s50i513_vn1300', 'jSO3'],
              ['UM_m01s50i514_vn1300', 'jMeONO2_CH3O_NO2'], # Matched to ATom
              ['UM_m01s50i515_vn1300', 'jNALD'],
              ['UM_m01s50i516_vn1300', 'jISON'],
              ['UM_m01s50i517_vn1300', 'EtONO2 -> NO2 + HO2 + MeCHO'], # Not matched to ATom because products are different.
              ['UM_m01s50i518_vn1300', 'jRN10NO3'],
              ['UM_m01s50i519_vn1300', 'i-PrONO2 J rate'], # isopropylONO2
              ['UM_m01s50i520_vn1300', 'jRN13NO3 --> MeCHO'],
              ['UM_m01s50i521_vn1300', 'jRN13NO3 --> CARB11A'],
              ['UM_m01s50i522_vn1300', 'jRN16NO3'], # Can be matched to jMeONO2?
              ['UM_m01s50i523_vn1300', 'jRN19NO3'],
              ['UM_m01s50i524_vn1300', 'jRA13NO3'],
              ['UM_m01s50i525_vn1300', 'jRA16NO3'],
              ['UM_m01s50i526_vn1300', 'jRA19NO3'],
              ['UM_m01s50i527_vn1300', 'jRTX24NO3'],
              ['UM_m01s50i528_vn1300', 'jRU10NO3'],
              ['UM_m01s50i529_vn1300', 'jRU14NO3'],
              ['UM_m01s50i530_vn1300', 'jCARB3 --> HO2'],
              ['UM_m01s50i531_vn1300', 'jCARB6'],
              ['UM_m01s50i532_vn1300', 'jCARB3 --> H2'],
              ['UM_m01s50i533_vn1300', 'jCARB3 --> HCHO'],
              ['UM_m01s50i540_vn1300', 'jMeCHO --> MeOO + HO2 + CO'],
              ['UM_m01s50i541_vn1300', 'jPropanal_CH2CH3_HCO'], # EtCHO matched to ATom. 
              ['UM_m01s50i542_vn1300', 'jHOCH2CHO'],
              ['UM_m01s50i543_vn1300', 'jUCARB12 --> MeCO3'],
              ['UM_m01s50i544_vn1300', 'jUCARB12 --> CARB7'],
              ['UM_m01s50i545_vn1300', 'jNUCARB12 --> HUCARB9'],
              ['UM_m01s50i546_vn1300', 'jTNCARB10'],
              ['UM_m01s50i547_vn1300', 'jRTN10OOH'],
              ['UM_m01s50i548_vn1300', 'jDHPCARB9'],
              ['UM_m01s50i549_vn1300', 'jHPUCARB12 --> HUCARB9'],
              ['UM_m01s50i550_vn1300', 'jHPUCARB12 --> CARB7'],
              ['UM_m01s50i551_vn1300', 'jHUCARB9'],
              ['UM_m01s50i552_vn1300', 'jDHPR12OOH'],
              ['UM_m01s50i553_vn1300', 'jDHCARB9'],
              ['UM_m01s50i554_vn1300', 'jNUCARB12 --> CARB7'],
              ['UM_m01s50i555_vn1300', 'jNO3_NO_O2'], # NO3 loss matched to ATom, preferred to 508.
              ['UM_m01s50i556_vn1300', 'peroxide DHPCARB9 J rate'],
              ['UM_m01s50i557_vn1300', 'jH2O'],
              ['UM_m01s50i558_vn1300', 'methane J rate'],
              ['UM_m01s50i559_vn1300', 'jHOBr_OH_Br'], # Matched to ATom
              ['UM_m01s50i560_vn1300', 'jHOCl'],
              ['UM_m01s50i561_vn1300', 'jHONO2'],
              ['UM_m01s50i562_vn1300', 'jHO2NO2'],
              ['UM_m01s50i563_vn1300', 'jH2O2_OH_OH'], # Matched to ATom
              ['UM_m01s50i564_vn1300', 'jMeOOH'], # MeOOH not matched to ATom CH3OOH because products are different.
              ['UM_m01s50i565_vn1300', 'jO2 --> O(3P)'],
              ['UM_m01s50i566_vn1300', 'jO2 --> O(1D)'],
              ['UM_m01s50i567_vn1300', 'jO3_O2_O1D'], # O3 -> O(1D) Matched to ATom
              ['UM_m01s50i568_vn1300', 'jN2O -> O(1D)'],
              ['UM_m01s50i569_vn1300', 'jMACR'], # MACR -> MeCO3 + HCHO + CO + HO2 Matched to ATom MVK or jMAC_NoProductsSpecified (possibly)
              ['UM_m01s50i570_vn1300', 'jMACROOH'],
              ['UM_m01s50i571_vn1300', 'jMeCHO --> CH4'],
              ['UM_m01s50i615_vn1300', 'jNRU12OOH a'],
              ['UM_m01s50i616_vn1300', 'jNRU12OOH b']]
