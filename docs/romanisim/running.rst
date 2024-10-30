Running the simulation
======================

The primary means by which we expect most users to make images is
the command line interface::

    romanisim-make-image out.asdf

The combination of ``romanisim-make-image`` and various user-generated
input catalogs allows most of the simulator functionality to be exercised [#chromatic]_.

.. highlight:: none

The ``romanisim-make-image`` command line interface (CLI) has a number of arguments to support
this functionality::

    romanisim-make-image -h
    usage: romanisim-make-image [-h] [--bandpass BANDPASS] [--boresight] [--catalog CATALOG] [--config CONFIG]
				[--date DATE] [--level LEVEL] [--ma_table_number MA_TABLE_NUMBER] [--nobj NOBJ]
				[--previous PREVIOUS] [--radec RADEC RADEC] [--rng_seed RNG_SEED] [--roll ROLL]
				[--sca SCA] [--usecrds] [--webbpsf] [--truncate TRUNCATE]
				[--pretend-spectral PRETEND_SPECTRAL] [--drop-extra-dq]
				filename

    Make a demo image.

    positional arguments:
      filename              output image (asdf)

    options:
      -h, --help            show this help message and exit
      --bandpass BANDPASS   bandpass to simulate (default: F087)
      --boresight           radec specifies location of boresight, not center of WFI. (default: False)
      --catalog CATALOG     input catalog (ecsv) (default: None)
      --config CONFIG       input parameter override file (yaml) (default: None)
      --date DATE           UTC Date and Time of observation to simulate in ISOT format. (default: None)
      --level LEVEL         1 or 2, for L1 or L2 output (default: 2)
      --ma_table_number MA_TABLE_NUMBER
      --nobj NOBJ
      --previous PREVIOUS   previous simulated file in chronological order used for persistence modeling. (default:
			    None)
      --radec RADEC RADEC   ra and dec (deg) (default: None)
      --rng_seed RNG_SEED
      --roll ROLL           Position angle (North towards YIdl) measured at the V2Ref/V3Ref of the aperture used.
			    (default: 0)
      --sca SCA             SCA to simulate. Use -1 to generate images for all SCAs; include {} in filename for this
			    mode to indicate where the SCA number should be filled, e.g. l1_wfi{}.asdf (default: 7)
      --usecrds             Use CRDS for distortion map (default: False)
      --webbpsf             Use webbpsf for PSF (default: False)
      --truncate TRUNCATE   If set, truncate the MA table at given number of resultants. (default: None)
      --pretend-spectral PRETEND_SPECTRAL
			    Pretend the image is spectral. exposure.type and instrument.element are updated to be
			    grism / prism. (default: None)
      --drop-extra-dq       Do not store the optional simulated dq array. (default: False)

    EXAMPLE: romanisim-make-image output_image.asdf

Expected arguments controlling things like the input :doc:`here </romanisim/catalog>` to
simulate, the right ascension and declination of the telescope
[#boresight]_, the :doc:`bandpass </romanisim/bandpass>`, the Sensor
Chip Assembly (SCA) to
simulate, the level of the image to simulate (:doc:`L1 </romanisim/l1>`
or :doc:`L2 </romanisim/l2>`), the MA table to use, and the time of
the observation.

Additional arguments control some details of the simulation.  The
``--usecrds`` argument indicates that reference files should be pulled
from the Roman CRDS server, and is recommended.  The ``--webbpsf``
argument indicates that the `WebbPSF
<https://webbpsf.readthedocs.io>`_ package should be used to simulate
the PSF; note that this presently disables chromatic PSF rendering.

The ``--rng_seed`` argument specifies a seed to the random number
generator, enabling reproducible results.

The ``--nobj`` argument is only used when a catalog is not specified,
and controls the number of objects that are simulated in that case.

The ``previous`` argument specifies the previous simulated frame.
This information is used to support :doc:`persistence </romanisim/l1>`
modeling.

.. [#chromatic] An important exception is the chromatic PSF rendering and 
   photon-shooting modes of GalSim; the current catalog format does 
   not support chromatic PSF rendering, and just assumes that all 
   sources are "gray" within a bandpass. 

.. [#boresight] This right ascension corresponds to either the
		location of the center of the WFI array or the
		telescope boresight, when the ``--boresight`` argument
		is specified.
