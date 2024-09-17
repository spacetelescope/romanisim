Point Spread Function Modeling
==============================

The simulator has two mechanisms for point source modeling.  The first uses the galsim implementation of the Roman point spread function (PSF); for more information, see the galsim Roman documentation.  The second uses the webbpsf package to make a model of the Roman PSF.

In the current implementation, the simulator uses a linearly varying, achromatic bandpass for each filter when using webbpsf.  That is, the PSF does not vary depending on the spectrum of the source being rendered.  However, it seems straightforward to implement either of these modes in the context of galsim, albeit at some computational expense.

When using the galsim PSF, galsim's "photon shooting" mode is used for efficient rendering of chromatic sources.  When using webbpsf, FFTs are used to do the convolution of the intrinsic source profile with the PSF and pixel grid of the instrument.

.. automodapi:: romanisim.psf

