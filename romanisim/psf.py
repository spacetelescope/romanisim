"""Roman PSF interface for galsim.

galsim.roman has an implementation of Roman's PSF based on the aperture and
some estimates for the wavefront errors over the aperture as described by
amplitudes of various Zernicke modes.  This seems like a very good approach,
but we want to add here a mode using the official PSFs coming out of
webbpsf, which takes a very similar overall approach.

galsim's InterpolatedImage class makes this straightforward.  Future work
should consider the following:

* how do we want to deal with PSF variation over the field?  Can we be more
  efficient than making a new InterpolatedImage at each location based on some
  e.g. quadratic approximation to the PSF's variation with location in an SCA?

* how do we want to deal with the dependence of the PSF on the source SED?
  It's possible I can just subclass ChromaticObject and implement
  evaluateAtWavelength, possibly also stealing the _shoot code from
  ChromaticOpticalPSF?

"""

import numpy as np
from astropy import units as u
import galsim
from galsim import roman
from .bandpass import galsim2roman_bandpass, roman2galsim_bandpass
from romanisim import log


def make_psf(sca, filter_name, wcs=None, webbpsf=True, pix=None,
             chromatic=False, **kw):
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
        function giving mapping from pixels to sky for use in computing local
        scale of image for webbpsf PSFs
    pix : tuple (float, float)
        pixel location of PSF on focal plane
    **kw : dict
        Additional keywords passed to galsim.roman.getPSF

    Returns
    -------
    profile : galsim.gsobject.GSObject
        galsim profile object for convolution with source profiles when rendering
        scenes.
    """
    pix = pix if pix is not None else (2044, 2044)
    if not webbpsf:
        filter_name = roman2galsim_bandpass[filter_name]
        defaultkw = {'pupil_bin': 8}
        if chromatic:
            defaultkw['n_waves'] = 10
            bandpass = None
        else:
            bandpass = roman.getBandpasses(AB_zeropoint=True)[filter_name]
            filter_name = None
        defaultkw.update(**kw)
        scapos = galsim.PositionD(*pix) if pix is not None else None
        return roman.getPSF(sca, filter_name, wcs=wcs, SCA_pos=scapos,
                            wavelength=bandpass, **defaultkw)
    if chromatic:
        log.warning('romanisim does not yet support chromatic PSFs '
                    'with webbpsf')
    import webbpsf as wpsf
    filter_name = galsim2roman_bandpass[filter_name]
    wfi = wpsf.WFI()
    wfi.detector = f'SCA{sca:02d}'
    wfi.filter = filter_name
    wfi.detector_position = pix
    oversample = kw.get('oversample', 4)
    psf = wfi.calc_psf(oversample=oversample)
    if wcs is None:
        scale = 0.11
    else:
        # get the actual pixel scale from the WCS
        cen = wcs.toWorld(galsim.PositionD(*pix))
        p1 = wcs.toWorld(galsim.PositionD(pix[0]+1, pix[1]))
        p2 = wcs.toWorld(galsim.PositionD(pix[0], pix[1]+1))
        scale = np.sqrt(cen.distanceTo(p1).deg*60*60 *
                        cen.distanceTo(p2).deg*60*60)
    intimg = galsim.InterpolatedImage(
        galsim.Image(psf[0].data, scale=scale / oversample),
        normalization='flux')
    return intimg
