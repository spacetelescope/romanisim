Overview
==================

romanisim simulates Roman Wide-Field Imager images of astronomical scenes
described by catalogs of sources.  The simulation includes:

* convolution of the sources by the Roman point spread function
* optical distortion
* sky background
* level 1 image support (3D image stack of up-the-ramp samples)
* level 2 image support (2D image after calibration & ramp fitting)
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
on top of the galsim.roman implementation are using "official" webbpsf
PSF, and distortion and reference files from the Roman CRDS (not yet
public!).  This
package also implements WFI up-the-ramp sampled and averaged images like those
that will be downlinked from the telescope, and the official Roman WFI file
format (asdf).

Future expected features include:

* frame zero effects
* L3 image simulations
* "image-based" simulation inputs (i.e., provide an input image based
  on a galaxy hydro sim; romanisim applies the Roman PSF &
  instrumental effects on top to produce a detailed instrumental simulation)

The best way to interact with romanisim is to make an image.  Running ::

    romanisim-make-image out.asdf

will make a test image in the file ``out.asdf``.  Naturally, usually one has a
particular astronomical scene in mind, and one can't really simulate a scene
without knowing where the telescope is pointing and when the observation is
being made.  A more complete invocation would be ::

    romanisim-make-image --catalog input.ecsv --radec 270 66 --bandpass F087 --sca 7 --date 2026 1 1 --level 1 out.asdf

where ``input.ecsv`` includes a list of sources in the scene, the
telescope boresight is pointing to (r, d) = (270, 66), the desired
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
      ├─seed (NoneType): None
      ├─simcatobj (NDArrayType): shape=(496,), dtype=void96
      ├─usecrds (bool): False
      └─webbpsf (bool): True

These fields are simply the arguments to ``romanisim-make-image``,
plus an additional ``simcatobj`` field with contains the ``x``, ``y``,
and number of photons of each simulated source.

Features not included so far:

* pedestal/frame 0 features
* non-linear dark features
