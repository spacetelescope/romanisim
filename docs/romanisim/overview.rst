romanisim overview
==================

romanisim simulates Roman Wide-Field Imager images of astronomical scenes
described by catalogs of sources.  The simulation includes:

* convolution of the sources by the appropriate PSF for each detector
* realistic world coordinate system
* sky background
* level 1 image support (3D image stack of up-the-ramp samples)
* level 2 image support (2D image after calibration & ramp fitting)
* point sources and analytic galaxy profiles
* expected system throughput
* dark current
* read noise
* inter-pixel capacitance
* non-linearity
* reciprocity failure

The simulator is based on galsim and most of these features directly invoke the
equivalents in the galsim.roman package.  The chief additions of this package
on top of the galsim.roman implementation are using "official" PSF, WCS, and
reference files from the Roman CRDS (not yet public!) and webbpsf.  This
package also implements WFI up-the-ramp sampled and averaged images like those
that will be downlinked from the telescope, and the official Roman WFI file
format (asdf).

The best way to interact with romanisim is to make an image.  Running

    romanisim-make-image out.asdf

will make a test image in the file `out.asdf`.  Naturally, usually one has a
particular astronomical scene in mind, and one can't really simulate a scene
without knowing where the telescope is pointing and when the observation is
being made.  A more complete invocation would be

    romanisim-make-image --catalog input.ecsv --radec 270 66 --bandpass F087 --sca 7 --date 2026 1 1 --level 1 out.asdf

where `input.ecsv` includes a list of sources in the scene, the telescope boresight is pointing to (r, d) = (270, 66), the desired bandpass is F087, the sensor is WFI07, the date is Jan 1, 2026, and a level 1 image (3D cube of samples up the ramp) is requested.

Features not included so far:

* any pedestal/frame 0 features
* non-linear dark features
* realistic treatment of the noise properties in L2 images, using information about how we go from L1 images & resultants to rate images.
