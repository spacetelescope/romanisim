# r0000101001001001001_0001_wfi01 - default for most steps
# r0000201001001001001_0001_wfi01 - equivalent for spectroscopic data
# r0000101001001001001_0002_wfi01 - a second resample exposure, only cal step needed
# r0000101001001001001_0003_wfi01 - for ramp fitting; truncated image
# r0000201001001001001_0003_wfi01 - for ramp fitting; truncated spectroscopic
#                                         we need only darkcurrent & ramp fit for these
#
# r0000101001001001001_0004_wfi01 - ma_table 110, 16 resultants file
# r0000201001001001001_0004_wfi01 - ma_table 110, 16 resultant file, spectroscopic

# note that the "spectroscopic" files are really imaging files where the only
# thing that has been updated is the optical_element and exposure.type.
# The MA tables are also supposed to be slightly different (frame time is different?)
# but I haven't done anything here.

# default image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:00:00 --rng_seed 1 --drop-extra-dq r0000101001001001001_0001_wfi01_uncal.asdf &
# different location, different exposure
romanisim-make-image --radec 270.00 66.01 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:05:00 --rng_seed 2 --drop-extra-dq r0000101001001001001_0002_wfi01_uncal.asdf &
# SCA 10
romanisim-make-image --radec 270.00 66.01 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:05:00 --rng_seed 2 --drop-extra-dq r0000101001001001001_0002_wfi10_uncal.asdf &
# default spectroscopic image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:10:00 --rng_seed 3 --drop-extra-dq r0000201001001001001_0001_wfi01_uncal.asdf --pretend-spectral GRISM &
# truncated image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:15:00 --rng_seed 4 --drop-extra-dq r0000101001001001001_0003_wfi01_uncal.asdf --truncate 6 &
# truncated spectroscopic image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:20:00 --rng_seed 5 --drop-extra-dq r0000201001001001001_0003_wfi01_uncal.asdf --truncate 6 --pretend-spectral GRISM &
# 16 resultant image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 110 --date 2027-06-01T00:25:00 --rng_seed 6 --drop-extra-dq r0000101001001001001_0004_wfi01_uncal.asdf &
# truncated spectroscopic image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --stpsf --usecrds --ma_table_number 110 --date 2027-06-01T00:30:00 --rng_seed 7 --drop-extra-dq r0000201001001001001_0004_wfi01_uncal.asdf --pretend-spectral GRISM &

wait
