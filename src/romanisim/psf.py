"""Roman PSF interface for galsim.

galsim.roman has an implementation of Roman's PSF based on the aperture and
some estimates for the wavefront errors over the aperture as described by
amplitudes of various Zernicke modes.  This seems like a very good approach,
but we want to add here a mode using the ``official'' PSFs coming out of
webbpsf, which takes a very similar overall approach.

galsim's InterpolatedImage class makes this straightforward.  Future work
should consider the following:
- how do we want to deal with PSF variation over the field?  Can we be more
  efficient than making a new InterpolatedImage at each location based on some
  e.g. quadratic approximation to the PSF's variation with location in an SCA?
- how do we want to deal with the dependence of the PSF on the source SED?
  It's possible I can just subclass ChromaticObject and implement
  evaluateAtWavelength, possibly also stealing the _shoot code from
  ChromaticOpticalPSF?
"""

import numpy as np
from astropy import units as u
import galsim
from galsim import roman


def make_psf(sca, filter_name, wcs=None, webbpsf=True, pix=None, **kw):
    """Make a PSF profile for Roman.

    Can construct both PSFs using galsim's built-in galsim.roman.roman_psfs
    routine, or can use webbpsf.

    Parameters
    ----------
    sca : int
        SCA number
    filter_name : str
        name of filter
    wcs : callable (optional)
        function giving mapping from pixels to sky for use in computing local scale
        of image for webbpsf PSFs
    pix : tuple (float, float)
        pixel location of PSF on focal plane
    **kw : dict
        Additional keywords passed to galsim.roman.getPSF

    Returns
    -------
    galsim profile object for convolution with source profiles when rendering
    scenes.
    """
    if not webbpsf:
        scapos = galsim.PositionD(*pix) if pix is not None else None
        return roman.getPSF(sca, filter_name, wcs=wcs, SCA_pos=scapos, **kw)
    import webbpsf as wpsf
    wfi = wpsf.WFI()
    wfi.detector = f'SCA{sca:02d}'
    wfi.filter = filter_name
    wfi.detector_position = pix if pix is not None else (2044, 2044)
    oversample = kw.get('oversample', 4)
    psf = wfi.calc_psf(oversample=oversample)
    if wcs is None:
        scale = 0.11
    else:
        # get the actual pixel scale from the WCS
        cen = wcs(pix[0], pix[1])
        p1 = wcs(pix[0]+1, pix[1])
        p2 = wcs(pix[0], pix[1]+1)
        scale = np.sqrt(p2.separation(cen).to(u.arcsec).value,
                        p1.separation(cen).to(u.arcsec).value)
    intimg = galsim.InterpolatedImage(
        galsim.Image(psf[0].data, scale=scale / oversample), normalization='flux')
    return intimg
