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
from collections import OrderedDict
from math import ceil

from astropy.convolution import Box2DKernel, convolve
from astropy.nddata import NDData
from astropy.time import Time
import numpy as np
import galsim
from galsim import roman
from photutils.psf import GriddedPSFModel

from roman_datamodels import datamodels
from .bandpass import galsim2roman_bandpass, roman2galsim_bandpass
from romanisim import log


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


def get_epsf_from_crds(sca, filter_name, date=None):
    """Retrieve EPSF reference model from CRDS

    Parameters
    ----------
    sca : int
        SCA number
    filter_name : str
        name of filter
    date : astropy.time.Time or None
        Date of simulation. If None, current date is used.

    Returns
    -------
    model : roman_datamodels.EpsfRefModel
    """
    from crds import getreferences

    if date is None:
        date = Time.now()
    header = {
        'ROMAN.META.INSTRUMENT.NAME': 'wfi',
        'ROMAN.META.INSTRUMENT.DETECTOR': f'SCA{sca:02d}',
        'ROMAN.META.INSTRUMENT.OPTICAL_ELEMENT': galsim2roman_bandpass[filter_name],
        'ROMAN.META.EXPOSURE.START_TIME': date.isot
    }
    ref_paths = getreferences(header, reftypes=['epsf'], observatory='roman')
    model = datamodels.open(ref_paths['epsf'])

    return model


def get_gridded_psf_model(psf_ref_model):
    """Function to generate gridded PSF model from psf reference file

    Compute a gridded PSF model for one SCA using the
    reference files in CRDS.
    The input reference files have 3 focus positions and this is using
    the in-focus images. There are also three spectral types that are
    available and this code uses the M5V spectal type.
    """
    # Open the reference file data model
    # select the infocus images (0) and we have a selection of spectral types
    # A0V, G2V, and M6V, pick G2V (1)
    focus = 0
    spectral_type = 1
    psf_images = psf_ref_model.psf[focus, spectral_type, :, :, :].copy()
    # get the central position of the cutouts in a list
    psf_positions_x = psf_ref_model.meta.pixel_x.data.data
    psf_positions_y = psf_ref_model.meta.pixel_y.data.data
    meta = OrderedDict()
    position_list = []
    for index in range(len(psf_positions_x)):
        position_list.append([psf_positions_x[index], psf_positions_y[index]])

    # integrate over the native pixel scale
    oversample = psf_ref_model.meta.oversample
    pixel_response_kernel = Box2DKernel(width=oversample)
    for i in range(psf_images.shape[0]):
        psf = psf_images[i, :, :]
        im = convolve(psf, pixel_response_kernel) * oversample**2
        psf_images[i, :, :] = im

    meta["grid_xypos"] = position_list
    meta["oversampling"] = oversample
    nd = NDData(psf_images, meta=meta)
    model = GriddedPSFModel(nd)

    return model


def make_one_psf(sca, filter_name, wcs=None, psftype='galsim', pix=None,
                 chromatic=False, oversample=4, extra_convolution=None, date=None, **kw):
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
    psftype : One of ['crds', 'galsim', 'stpsf']
        How to determine the PSF.
    pix : tuple (float, float)
        pixel location of PSF on focal plane
    chromatic : bool
        Create a multiwavelength-based psf.
    oversample : int
        oversampling with which to sample Stpsf PSF
    extra_convolution : galsim.gsobject.GSObject or None
        Additional convolution to add to PSF
    date : astropy.time.Time or None
        Date of simulation. If None, current date is used. Needed for psftype='crds'
        to choose the appropriate epsf reference.
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

    # Create the PSF depending on method desired.
    if psftype == 'stpsf':
        psf = make_one_psf_stpsf(sca, filter_name, wcs=wcs, pix=pix, chromatic=chromatic,
                                 oversample=oversample, extra_convolution=extra_convolution, **kw)
    elif psftype == 'crds':
        psf = make_one_psf_crds(sca, filter_name, wcs=wcs, pix=pix, chromatic=chromatic,
                                extra_convolution=extra_convolution, date=date, **kw)
    else:  # Default is galsim
        psf = make_one_psf_galsim(sca, filter_name, wcs=wcs, pix=pix, chromatic=chromatic, extra_convolution=extra_convolution, **kw)

    return psf


def make_one_psf_crds(sca, filter_name, wcs=None, pix=None,
                      chromatic=False, extra_convolution=None, date=None, **kw):
    """Make a PSF profile for Roman at a specific detector location using CRDS reftype epsf

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
    chromatic : bool
        Create a multiwavelength-based psf.
    extra_convolution : galsim.gsobject.GSObject or None
        Additional convolution to add to PSF
    date : astropy.time.Time or None
        Date of simulation. If None, current date is used. Needed for psftype='crds'
        to choose the appropriate epsf reference.
    **kw : dict
        Additional keywords passed to galsim.roman.getPSF or stpsf.calc_psf,
        depending on whether stpsf is set.

    Returns
    -------
    profile : galsim.gsobject.GSObject
        galsim profile object for convolution with source profiles when
        rendering scenes.
    """
    log.info('Creating PSF from CRDS reference type epsf')
    if chromatic:
        log.warning('romanisim does not yet support chromatic PSFs '
                    'with stpsf')
    epsf_ref_model = get_epsf_from_crds(sca, filter_name, date=date)
    gridded_psf = get_gridded_psf_model(epsf_ref_model)

    psf = psf_from_grid(gridded_psf, *pix)
    intimg = psf_to_galsimimage(psf, wcs=wcs, pix=pix, oversample=gridded_psf.meta['oversampling'],
                                pixelscale=1., extra_convolution=extra_convolution)
    return intimg


def make_one_psf_galsim(sca, filter_name, wcs=None, pix=None,
                        chromatic=False, extra_convolution=None, **kw):
    """Make a PSF profile for Roman at a specific detector location using the galsim library

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
    log.info('Creating PSF using galsim')
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


def make_one_psf_stpsf(sca, filter_name, wcs=None, pix=None,
                       chromatic=False, oversample=4, extra_convolution=None, **kw):
    """Make a PSF profile for Roman at a specific detector location using the galsim library

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
    chromatic : bool
        Create a multiwavelength-based psf.
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
    log.info('Creating PSF using stpsf')
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
    intimg = psf_to_galsimimage(psf[0].data, wcs=wcs, pix=pix, oversample=oversample,
                                pixelscale=wfi.pixelscale, extra_convolution=extra_convolution)
    return intimg


def make_psf(sca, filter_name, wcs=None, psftype='galsim', pix=None,
             chromatic=False, variable=False, extra_convolution=None, date=None, **kw):
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
    psftype : One of ['crds', 'galsim', 'stpsf]
        How to determine the PSF.
    pix : tuple (float, float)
        pixel location of PSF on focal plane
    variable : bool
        True if a variable PSF object is desired
    date : astropy.time.Time or None
        Date of simulation. If None, current date is used. Needed for psftype='crds'
        to choose the appropriate epsf reference.
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
        return make_one_psf(sca, filter_name, wcs=wcs, psftype=psftype,
                            pix=pix, chromatic=chromatic,
                            extra_convolution=extra_convolution, date=date, **kw)
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
        psfs[corner] = make_one_psf(sca, filter_name, wcs=wcs, psftype=psftype,
                                    pix=pix, chromatic=chromatic,
                                    extra_convolution=extra_convolution, **kw)
    return VariablePSF(corners, psfs)


def pix_coords(x_start=0, x_end=5, y_start=0, y_end=5, n_x=None, n_y=None):
    """Generate coordinate arrays covering the specified domain

    Create array X and Y such that the corresponding values of each
    array creates a coordinate, (X[idx], Y[idx]) within the specified
    domain

    Parameters
    ----------
    x_start, x_end, y_start, y_end : float
        The domain to be covered. The condition start < end should be satisfied.

    Returns
    -------
    x, y : np.array, np.array
        The X and Y coordinates arrays.

    Examples
    --------
    >>> x, y = pix_coords()
    >>> print(x)
    [[0.   1.25 2.5  3.75 5.  ]
    [0.   1.25 2.5  3.75 5.  ]
    [0.   1.25 2.5  3.75 5.  ]
    [0.   1.25 2.5  3.75 5.  ]
    [0.   1.25 2.5  3.75 5.  ]]
    >>> print(y)
    [[0.   0.   0.   0.   0.  ]
    [1.25 1.25 1.25 1.25 1.25]
    [2.5  2.5  2.5  2.5  2.5 ]
    [3.75 3.75 3.75 3.75 3.75]
    [5.   5.   5.   5.   5.  ]]
    """
    if n_x is None:
        n_x = ceil(x_end - x_start)
    if n_y is None:
        n_y = ceil(y_end - y_start)
    rx = np.array([np.linspace(x_start, x_end, n_x)])
    x = rx.repeat(n_y, axis=0)
    ry = np.array([np.linspace(y_start, y_end, n_y)]).T
    y = ry.repeat(n_x, axis=1)

    return x, y


def psf_from_grid(psfgrid, x_0=None, y_0=None):
    """Calculate a PSF profile from a GriddedPSFModel at the specified position

    Parameters
    ----------
    psfgrid : GriddedPSFModel
        The PSF model to calculate from

    x_0, y_0 : float or None
        Position to calculate the psf. If None, (0., 0.) is used

    Returns
    -------
    psf : nd.array
        The psf profile.
    """
    x_0 = 0. if x_0 is None else x_0
    y_0 = 0. if y_0 is None else y_0
    bb = psfgrid.get_bounding_box()
    x, y = pix_coords(x_0 + bb.intervals[0].lower, x_0 + bb.intervals[0].upper,
                      y_0 + bb.intervals[1].lower, y_0 + bb.intervals[1].upper)
    psf = psfgrid.evaluate(x, y, 1, x_0, y_0)
    return psf


def psf_to_galsimimage(psf, wcs=None, pix=None, oversample=4, pixelscale=1., extra_convolution=None):
    """Convert an STPSF/CRDS PSF profile to galsim.Image"""

    # stpsf doesn't do distortion
    # calc_psf gives something aligned with the pixels, but with
    # a constant pixel scale equal to wfi.pixelscale / oversample.
    # we need to get the appropriate rotated WCS that matches this
    newscale = pixelscale / oversample
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
    gimg = galsim.Image(psf, wcs=newwcs)

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
