Overview
==================

romanisim simulates Roman Wide-Field Imager images of astronomical scenes
described by catalogs of sources.  The simulation includes:

* convolution of the sources by the Roman point spread function (PSF)
* optical distortion
* sky background
* level 1 image support (L1, 3D image stack of up-the-ramp samples)
* level 2 image support (L2, 2D image after calibration & ramp fitting)
* level 3 image support (L3, approximate coadded images)
* point sources and analytic galaxy profiles
* expected system throughput
* dark current
* read noise
* inter-pixel capacitance
* non-linearity
* saturation
* ramp fitting
* source injection

The simulator is based on galsim and most of these features directly invoke the
equivalents in the galsim.roman package.  The chief additions of this package
on top of the galsim.roman implementation are using "official" stpsf
PSFs, and many reference files from the Roman `CRDS <https://roman-crds.stsci.edu>`_ CRDS.  This
package also implements WFI up-the-ramp sampled and averaged images like those
that will be downlinked from the telescope, and the official Roman WFI file
format (asdf).

The best way to interact with romanisim is to make an image.  Running ::

    romanisim-make-image out.asdf

will make a test image in the file ``out.asdf``.  Naturally, usually one has a
particular astronomical scene in mind, and one can't really simulate a scene
without knowing where the telescope is pointing and when the observation is
being made.  A more complete invocation would be ::

    romanisim-make-image --catalog input.ecsv --radec 270 66 --bandpass F087 --sca 7 --date 2026 1 1 --level 1 out.asdf

where ``input.ecsv`` includes a list of sources in the scene, the
telescope boresight is pointing to (RA, dec) = (270, 66), the desired
bandpass is F087, the sensor is WFI07, the date is Jan 1, 2026, and a
level 1 image (3D cube of samples up the ramp) is requested.

The output of ``romanisim-make-image`` is an appropriate asdf file for
the requested level image, with the following addition.  The script
adds an additional top-level branch to the asdf tree with the name
``romanisim``.  Here's an example::

    └─romanisim (dict)
      ├─bandpass (str): F087
      ├─catalog (NoneType): None
      ├─date (NoneType): None
      ├─filename (str): out.asdf
      ├─level (int): 1
      ├─ma_table_number (int): 1
      ├─radec (NoneType): None
      ├─sca (int): 7
      ├─rng_seed (NoneType): None
      ├─simcatobj (NDArrayType): shape=(496,), dtype=void96
      ├─usecrds (bool): False
      └─stpsf (bool): True

These fields are simply the arguments to ``romanisim-make-image``,
plus an additional ``simcatobj`` field which contains the ``x``, ``y``,
and number of photons of each simulated source.

Features not included so far:

* pedestal/frame 0 features
* non-linear dark features
