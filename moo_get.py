from datetime import datetime, timedelta

year = 2016
start_date = datetime(year, 1, 1)
end_date = datetime(year + 1, 1, 1)
current_date = start_date

moo_get_cmds, rsync_cmds = [], []
while current_date < end_date:
  date = current_date.strftime('%m%d')
  filename = f'cy731a.pl2016{date}.pp'
  get = (f'moo get -v moose:/crum/u-cy731/apl.pp/{filename} .')
  pull = (f'rsync -v sophiet@xfer-vm-01.jasmin.ac.uk:~/{filename} /scratch/st838/netscratch/data/ukca_pps/')
  moo_get_cmds.append(get)
  rsync_cmds.append(pull)
  current_date += timedelta(days=1)
  
max_files = 4  
  
for i in range(0, len(moo_get_cmds)-max_files, max_files):
  print()
  print('rm cy731a.* MetOffice*')
  for j in range(max_files):
    print(moo_get_cmds[i+j])
  print()
  for j in range(max_files):
    print(rsync_cmds[i+j])
  print()
