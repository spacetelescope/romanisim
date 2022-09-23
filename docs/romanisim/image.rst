Making Images
=============

Ultimately, romanisim builds image by feeding source profiles, world coordinate system objects, and point spread functions to galsim.  The `image` and `l1` modules currently implement this functionality.

The `image` module is responsible for translating a metadata object that specifies everything about the conditions of the observation into objects that the simulation can understand.  The metadata object follows the metadata that real WFI images will include; see [here](https://roman-pipeline.readthedocs.io/en/latest/roman/datamodels/metadata.html#metadata) for more information.

The parsed metadata is used to make a `counts` image that is an idealized image containing the number of photons each WFI pixel would collect over an observation.  It includes no systematic effects or noise beyond Poisson noise from the astronomical scene and backgrounds.  Actual WFI observations are more complicated than just noisy versions of this idealized image, however, for several reasons:

* WFI pixels have a very uncertain pedestal.
* WFI pixels are sampled "up the ramp" during an observation, so a number of reads contribute to the final estimate for the rate of photons entering each pixel.
* WFI reads are averaged on the telescope into resultants; ground images see only resultants.

These idealized count images are then used to either make a level 2 image or a level 1 image, which are intended to include
the effects of these complications.  The construction of L1 images is described :doc:`here </romanisim/l1>`.

L2 images are currently constructed from count images at present simply by adding the following effects: reciprocity failure, nonlinearity, interpixel capacitance, and read noise.  Then the gain is divided out and the image is quantized.  The L2 images also do not yet remove the mean dark count rate.  Most critically, this process ignores the details of ramp fitting and how that translates into uncertainties on each pixel, but at least the pieces are in place to include that sort of effect.  The L2 images should not presently be taken seriously.



