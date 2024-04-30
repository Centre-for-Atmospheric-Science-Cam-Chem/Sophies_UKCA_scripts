Depending on the suite run, it should currently be using one of the .dat files whose names begin with ‘FJX_spec’, which can be found at https://code.metoffice.gov.uk/trac/um/browser/aux/trunk/ctldata/UKCA/fastj

The UKCA file here is different because its HCHO data has been updated using the GEOS-chem file.

Place the UKCA file somewhere and point to it in your Rose suite, either in the Rose GUI or in the suite's app/um/rose-app.conf file. 
e.g.
jvspec_dir='/home/d02/sturner/fastJ'
jvspec_file='UKCA_FJX.dat'

