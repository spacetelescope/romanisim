# r0000101001001001001_0001_WFI01 - default for most steps
# r0000201001001001001_0001_WFI01 - equivalent for spectroscopic data
# r0000101001001001001_0002_WFI01 - a second resample exposure, only cal step needed
# r0000101001001001001_0003_WFI01 - for ramp fitting; truncated image
# r0000201001001001001_0003_WFI01 - for ramp fitting; truncated spectroscopic
#                                         we need only darkcurrent & ramp fit for these
#
# r0000101001001001001_0004_WFI01 - 16 resultant file, ma_table 110
# r0000201001001001001_0004_WFI01 - 16 resultant spectroscopic file, ma_table 110

# note that the "spectroscopic" files are really imaging files where the only
# thing that has been updated is the optical_element and exposure.type.
# The MA tables are also supposed to be slightly different (frame time is different?)
# but I haven't done anything here.

# default image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:00:00 --rng_seed 1 --drop-extra-dq r0000101001001001001_0001_{}_uncal.asdf &
# second image at a different location, different activity
romanisim-make-image --radec 270.00 66.01 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:05:00 --rng_seed 2 --drop-extra-dq r0000101001001001001_0002_{}_uncal.asdf &
# default spectroscopic image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:10:00 --rng_seed 3 --drop-extra-dq r0000201001001001001_0001_{}_uncal.asdf --pretend-spectral GRISM &
# truncated image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:15:00 --rng_seed 4 --drop-extra-dq r0000101001001001001_0003_{}_uncal.asdf --truncate 6 &

# just do four at a time
wait

# truncated spectroscopic image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:20:00 --rng_seed 5 --drop-extra-dq r0000201001001001001_0003_{}_uncal.asdf --truncate 6 --pretend-spectral GRISM &
# 16 resultant image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 110 --date 2027-06-01T00:25:00 --rng_seed 6 --drop-extra-dq r0000101001001001001_0004_{}_uncal.asdf &
# truncated spectroscopic image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 110 --date 2027-06-01T00:30:00 --rng_seed 7 --drop-extra-dq r0000201001001001001_0004_{}_uncal.asdf --pretend-spectral GRISM &

# prism image
romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --webbpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:30:00 --rng_seed 8 --drop-extra-dq r0000301001001001001_0001_{}_uncal.asdf --pretend-spectral PRISM &

wait
