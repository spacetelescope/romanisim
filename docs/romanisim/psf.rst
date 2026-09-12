Point Spread Function Modeling
==============================

The simulator has two mechanisms for point source modeling.  The first uses the galsim implementation of the Roman point spread function (PSF); for more information, see the galsim Roman documentation.  The second uses the stpsf package to make a model of the Roman PSF.

In the current implementation, the simulator uses a linearly varying, achromatic bandpass for each filter when using stpsf.  That is, the PSF does not vary depending on the spectrum of the source being rendered.  However, it seems straightforward to implement either of these modes in the context of galsim, albeit at some computational expense.

When using the galsim PSF, galsim's "photon shooting" mode is used for efficient rendering of chromatic sources.  When using stpsf, FFTs are used to do the convolution of the intrinsic source profile with the PSF and pixel grid of the instrument.

A third mechanism, ``--psftype epsf``, takes the PSF from the CRDS ``epsf``
reference files.  Two conventions for those files are supported, and
romanisim tells them apart by their normalization; see
:func:`romanisim.psf.epsf_is_pixel_convolved`.

Older references hold the optical PSF on an idealized grid of square 0.11 arcsecond
pixels, normalized so that the stamp sums to the enclosed flux fraction, and
without convolution by the pixel response function.
romanisim renders these by convolving with the pixel grid of the instrument.

Newer reference files hold more empirical PSFs: on native pixels, and convolved
with the pixel response
function, with IPC already applied.  These are
normalized so that taking every ``oversample``-th sample gives exactly the
fraction of the flux landing in real native pixels at that subpixel offset,
which makes the whole stamp sum to about ``oversample ** 2``.  For these
PSFs romanisim

* deconvolves the IPC, so that when constructing an L1 it can be reapplied,
  correlating Poisson noise as will be present in real images;
* attaches the local WCS to the stamp so that the distortion is properly tracked
* renders the PSF with GalSim's ``no_pixel`` mode, since pixel convolution has
  already been applied.

Because these empirical PSFs already include the pixel response, they are incompatible
with photon shooting and chromatic rendering is not available for them.

.. automodapi:: romanisim.psf

