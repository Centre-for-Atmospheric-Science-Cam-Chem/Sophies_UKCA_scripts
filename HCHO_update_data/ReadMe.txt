Depending on the suite run, the suite's Fast-J spectral data file should currently be using one of the .dat files whose names begin with ‘FJX_spec’, which can be found at https://code.metoffice.gov.uk/trac/um/browser/aux/trunk/ctldata/UKCA/fastj

The UKCA new file here has had its HCHO data has been updated using the GEOS-chem file.

Place the UKCA file somewhere and point to it in your Rose suite, either in the Rose GUI or in the suite's app/um/rose-app.conf file. 
e.g.
jvspec_dir='/home/d02/sturner/fastJ'
jvspec_file='UKCA_FJX_new.dat'

