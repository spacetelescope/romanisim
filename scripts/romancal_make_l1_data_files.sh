# r0000101001001001001_0001_wfi**_f158 - default for most steps, ma table 109
# r0000201001001001001_0001_wfi01_f158 - equivalent for spectroscopic data
# r0000301001001001001_0001_wfi01_f158 - prism image, ma table 109

# note that the "spectroscopic" files are really imaging files where the only
# thing that has been updated is the optical_element and exposure.type.
# The MA tables are also supposed to be slightly different (frame time is different?)
# but I haven't done anything here.

# stop on an error
set -e

# default image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --psftype stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:00:00 --rng_seed 1 --drop-extra-dq r0000101001001001001_0001_{}_{bandpass}_uncal.asdf

# default spectroscopic image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --psftype stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:10:00 --rng_seed 3 --drop-extra-dq r0000201001001001001_0001_{}_{bandpass}_uncal.asdf --pretend-spectral GRISM
# 
# prism image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --psftype stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:30:00 --rng_seed 8 --drop-extra-dq r0000301001001001001_0001_{}_{bandpass}_uncal.asdf --pretend-spectral PRISM
