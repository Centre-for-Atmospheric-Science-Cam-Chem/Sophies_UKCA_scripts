'''
Date: 21/7/2024
Contact: st838@cam.ac.uk
List of file paths to import into scripts.
For use on atm-farman.
'''

base = '/scratch/st838/netscratch'
data = f'{base}/data'
mod = f'{base}/models'
analysis = f'{base}/analysis'
code = f'{base}/scripts'
ATom = f'{data}/ATom_MER10_Dataset'
ATom_csv = f'{ATom}/ATom_hourly_all.csv'
UKCA_csv = f'{data}/nudged_J_outputs_for_ATom/UKCA_hourly_all.csv'
npy = f'{data}/ukca_npys'
year = f'{npy}/low_res_yr_500k.npy'
test_year = f'{npy}/low_res_yr_50.npy'
pp = f'{data}/ukca_pps'
rf = f'{mod}/rf'
rf_scaled = f'{mod}/rf_scaled'
con = f'{code}/constants.py'
