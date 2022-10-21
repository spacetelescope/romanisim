Reference files
===============

romanisim uses reference files from `CRDS <https://hst-crds.stsci.edu/static/users_guide/index.html>`_ in order to simulate realistic images.  The following kinds of reference files are used:

* read noise
* dark current
* flat field
* gain
* distortion map

The usage of these is mostly straightforward, but we provide here a few notes.

Read Noise
----------
The random noise on reading a sample contributing to a ramp in an L1 image is scaled by the read noise reference file.

Dark Current
------------
CRDS provides dark current images for each possible MA table, including the averaging of the dark current into resultants.  This simplifies subtraction from L1 images and allows effects beyond a simple Poisson sampling of dark current electrons in each read.  But it's unwieldy for a simulator because any effects beyond simple Poisson sampling of dark current electrons are not presently defined well enough to allow simulation.  So the simulator simply takes the last resultant in the dark current resultant image and scales it by the effective exposure time of that resultant to get a dark current rate.  This rate then goes into the idealized "counts" image which is then apportioned into the reads making up the resultants of an L1 image.

Flat field
----------
Implementation of the flat field requires a little care due to the desire to support galsim's "photon shooting" rendering mode.  This mode does not create noise-free images but instead only simulates the number of photons that would be actually detected in a device.  We want to start by simulating the number of photons each pixel would record for a flat field of 1, and then sample that down by a fraction corresponding to the actual QE of each pixel.  That works fine supposing that the flat field is less than 1, but does not work for values of the flat field greater than 1.  So we instead do the initial galsim simulations for a larger exposure time than necessary, scaled by the maximum value of the flat field, and then sample down by :math:`\mathrm{flat}/\mathrm{maxflat}`.  That's all well and good as long as there aren't any spurious very large values in the flat field.  I haven't actually seen any such values yet and so haven't tried to address that case (e.g., by clipping them).

Gain
----
Photons from the idealized "counts" image are scaled down to ADU before quantization during L1 creation, and then converted back to electrons before ramp fitting when making L2 images.

Distortion map
--------------
World coordinate systems for WFI images are created by placing the telescope boresight at :math:`\mathrm{V2} = \mathrm{V3} = 0`, and then applying the distortion maps from CRDS to convert from V2V3 to pixels.
