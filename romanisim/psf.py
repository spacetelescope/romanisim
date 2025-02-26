"""Roman point spread function (PSF) interface for galsim.

galsim.roman has an implementation of Roman's point spread function (PSF) based on
the aperture and some estimates for the wavefront errors over the aperture as
described by amplitudes of various Zernicke modes.  This seems like a very good
approach, but we want to add here a mode using the official PSFs coming out of
stpsf, which takes a very similar overall approach.

galsim's InterpolatedImage class makes this straightforward.  Future work
should consider the following:

* How do we want to deal with the dependence of the PSF on the source SED?
  It's possible we can just subclass ChromaticObject and implement
  evaluateAtWavelength, using the _shoot code from ChromaticOpticalPSF.
"""

import numpy as np
import galsim
from galsim import roman
from .bandpass import galsim2roman_bandpass, roman2galsim_bandpass
from romanisim import log


def make_one_psf(sca, filter_name, wcs=None, stpsf=True, pix=None,
                 chromatic=False, oversample=4, extra_convolution=None, **kw):
    """Make a PSF profile for Roman at a specific detector location.

    Can construct both PSFs using galsim's built-in galsim.roman.roman_psfs
    routine, or can use stpsf.

    Parameters
    ----------
    sca : int
        SCA number
    filter_name : str
        name of filter
    wcs : callable (optional)
        function giving mapping from pixels to sky for use in computing local
        scale of image for stpsf PSFs
    pix : tuple (float, float)
        pixel location of PSF on focal plane
    oversample : int
        oversampling with which to sample Stpsf PSF
    extra_convolution : galsim.gsobject.GSObject or None
        Additional convolution to add to PSF
    **kw : dict
        Additional keywords passed to galsim.roman.getPSF or stpsf.calc_psf,
        depending on whether stpsf is set.

    Returns
    -------
    profile : galsim.gsobject.GSObject
        galsim profile object for convolution with source profiles when
        rendering scenes.
    """
    pix = pix if pix is not None else (roman.n_pix // 2, roman.n_pix // 2)
    if wcs is None:
        log.warning('wcs is None; unlikely to get orientation of PSF correct.')
    if not stpsf:
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
        res = roman.getPSF(sca, filter_name, wcs=wcs, SCA_pos=scapos,
                           wavelength=bandpass, **defaultkw)
        if extra_convolution is not None:
            res = galsim.Convolve(res, extra_convolution)
        return res
    if chromatic:
        log.warning('romanisim does not yet support chromatic PSFs '
                    'with stpsf')
    import stpsf as wpsf
    filter_name = galsim2roman_bandpass[filter_name]
    wfi = wpsf.WFI()
    wfi.detector = f'SCA{sca:02d}'
    wfi.filter = filter_name
    wfi.detector_position = pix
    psf = wfi.calc_psf(oversample=oversample, **kw)
    # stpsf doesn't do distortion
    # calc_psf gives something aligned with the pixels, but with
    # a constant pixel scale equal to wfi.pixelscale / oversample.
    # we need to get the appropriate rotated WCS that matches this
    newscale = wfi.pixelscale / oversample
    if wcs is not None:
        local_jacobian = wcs.local(image_pos=galsim.PositionD(pix)).getMatrix()
        # angle of [du/dx, du/dy]
        ang = np.arctan2(local_jacobian[0, 1], local_jacobian[0, 0])
        rotmat = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
        newwcs = galsim.JacobianWCS(*(rotmat.ravel() * newscale))
        # we are making a new, orthogonal, isotropic matrix for the PSF with the
        # appropriate pixel scale.  This is intended to be the WCS for the PSF
        # produced by stpsf.
    else:
        newwcs = galsim.JacobianWCS(*(np.array([1, 0, 0, 1]) * newscale))
        # just use a default North = up WCS
    gimg = galsim.Image(psf[0].data, wcs=newwcs)

    # This code block could be used to fix the centroid of Stpsf calculated
    # PSFs to be zero.  This makes downstream comparisons with Stpsf
    # PSFs a little harder, and so is currently disabled.  But it is
    # recommended by Marshall Perrin and is probably what we should do.

    #  centroid = []
    #  for i, ll in enumerate(psf[0].data.shape):
    #      cc = np.arange(ll) - (ll - 1) / 2
    #      newshape = [1] * len(psf[0].data.shape)
    #      newshape[-(i + 1)] = -1
    #      cen = np.sum(cc.reshape(newshape) * psf[0].data) / np.sum(psf[0].data)
    #      centroid.append(cen)
    #  centroid = np.array(centroid)

    centroid = None
    intimg = galsim.InterpolatedImage(
        gimg, normalization='flux', use_true_center=True, offset=centroid)

    if extra_convolution is not None:
        intimg = galsim.Convolve(intimg, extra_convolution)

    return intimg


def make_psf(sca, filter_name, wcs=None, stpsf=True, pix=None,
             chromatic=False, variable=False, extra_convolution=None, **kw):
    """Make a PSF profile for Roman.

    Optionally supports spatially variable PSFs via interpolation between
    the four corners of an SCA.

    Parameters
    ----------
    sca : int
        SCA number
    filter_name : str
        name of filter
    wcs : callable (optional)
        function giving mapping from pixels to sky for use in computing local
        scale of image for stpsf PSFs
    pix : tuple (float, float)
        pixel location of PSF on focal plane
    variable : bool
        True if a variable PSF object is desired
    extra_convolution : galsim.gsobject.GSObject or None
        Additional convolution to add to PSF profiles
    **kw : dict
        Additional keywords passed to make_one_psf

    Returns
    -------
    profile : galsim.gsobject.GSObject
        galsim profile object for convolution with source profiles when
        rendering scenes.
    """
    if not variable:
        return make_one_psf(sca, filter_name, wcs=wcs, stpsf=stpsf,
                            pix=pix, chromatic=chromatic,
                            extra_convolution=extra_convolution, **kw)
    elif pix is not None:
        raise ValueError('cannot set both pix and variable')
    buf = 49
    # Stpsf complains if we get too close to (0, 0) for some reason.
    # For other corners one can go to within a fraction of a pixel.
    # if we go larger than 49 we have to change some of the tests, which use a 100x100 image.
    corners = dict(
        ll=[buf, buf], lr=[roman.n_pix - buf, buf],
        ul=[buf, roman.n_pix - buf], ur=[roman.n_pix - buf, roman.n_pix - buf])
    psfs = dict()
    for corner, pix in corners.items():
        psfs[corner] = make_one_psf(sca, filter_name, wcs=wcs, stpsf=stpsf,
                                    pix=pix, chromatic=chromatic,
                                    extra_convolution=extra_convolution, **kw)
    return VariablePSF(corners, psfs)


class VariablePSF:
    """Spatially variable PSF wrapping GalSim profiles.

    Linearly interpolates between four corner PSF profiles by summing
    weighted GalSim PSF profiles.
    """

    def __init__(self, corners, psf):
        self.corners = corners
        self.psf = psf

    def at_position(self, x, y):
        """Instantiate a PSF profile at (x, y).

        Linearly interpolate between the four corners to obtain the
        PSF at this location.

        Parameters
        ----------
        x : float
            x position
        y : float
            y position

        Returns
        -------
        GalSim profile representing PSF at (x, y).
        """
        npix = self.corners['ur'][-1]
        off = self.corners['ll'][0]
        wleft = np.clip((npix - x) / (npix - off), 0, 1)
        wlow = np.clip((npix - y) / (npix - off), 0, 1)
        # x = [0, off] -> 1
        # x = [npix, infinity] -> 0
        # linearly between those, likewise for y.
        out = (self.psf['ll'] * wleft * wlow
               + self.psf['lr'] * (1 - wleft) * wlow
               + self.psf['ul'] * wleft * (1 - wlow)
               + self.psf['ur'] * (1 - wleft) * (1 - wlow))
        return out
