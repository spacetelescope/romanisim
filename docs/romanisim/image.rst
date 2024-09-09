Making images
=============

Ultimately, romanisim builds image by feeding source profiles, world coordinate system objects, and point spread functions to galsim.  The ``image`` and ``l1`` modules currently implement this functionality.

The ``image`` module is responsible for translating a metadata object that specifies everything about the conditions of the observation into objects that the simulation can understand.  The metadata object follows the metadata that real WFI images will include; see `here <https://roman-pipeline.readthedocs.io/en/latest/roman/datamodels/metadata.html#metadata>`_ for more information.

The parsed metadata is used to make a ``counts`` image that is an idealized image containing the number of photons each WFI pixel would collect over an observation.  It includes no systematic effects or noise beyond Poisson noise from the astronomical scene and backgrounds.  Actual WFI observations are more complicated than just noisy versions of this idealized image, however, for several reasons:

* WFI pixels have a very uncertain pedestal.
* WFI pixels are sampled "up the ramp" during an observation, so a number of reads contribute to the final estimate for the rate of photons entering each pixel.
* WFI reads are averaged on the telescope into resultants; ground images see only resultants.

These idealized count images are then used to either make a level 2 image or a level 1 image, which are intended to include
the effects of these complications.  The construction of L1 images is described :doc:`here </romanisim/l1>`, and the construction of L2 images is described :doc:`here </romanisim/l2>`.

L2 source injection
===================

We support injecting sources into L2 files.  L2 source injection follows many of the same steps as L1 creation, with some short cuts.  The simulation creates an idealized image of the total number of counts deposited into each pixel, using the provided catalog, exposure time, filter, WCS, and PSF from the provided image.  A virtual ramp is generated for the existing L2 file by evenly apportioning the L2 flux among the resultants.  Additional counts from the injected source are added to each resultant in the virtual ramp using the same code as for the L1 simulations.  The resulting virtual ramp is fit to get the new flux per pixel, and we replace the values in the original L2 model with the new slope measurements.

This makes some shortcuts:

* The L2 files don't include information about which pixels contain CR hits.  Sources injected into pixels with CR hits get re-fit as if there were no CR.
* Non-linearity is not included; the ramp is refit without any non-linearity.
* The ramp is refit without persistence, but any persistence which was included in the initial ramp slope is still included.

The function :meth:`romanisim.image.inject_sources_into_l2` is the intended entry point for L2 source injection; see its documentation for more details about how to add sources to an L2 image.



.. automodapi:: romanisim.image
