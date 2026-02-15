from collections import OrderedDict
from functools import cache

import galsim
import numpy as np

from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel
from roman_datamodels import datamodels
from scipy import interpolate

from romanisim import log

# from .bandpass import galsim2roman_bandpass, roman2galsim_bandpass, getBandpasses
from .bandpass import getBandpasses
from .parameters import (
    default_date,
    galsim2roman_bandpass,
    n_pix,
    pixel_scale,
    roman2galsim_bandpass,
)
from .psf_utils import getPSF


class VariablePSF:
    """Spatially variable PSF wrapping GalSim profiles.

    Linearly interpolates between four corner PSF profiles by summing
    weighted GalSim PSF profiles.
    """

    def __init__(self, corners, psf):
        self.corners = corners
        self.psf = psf
        self.psfinterpolators = None

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
        npix = self.corners["ur"][-1]
        off = self.corners["ll"][0]
        wleft = np.clip((npix - x) / (npix - off), 0, 1)
        wlow = np.clip((npix - y) / (npix - off), 0, 1)
        # x = [0, off] -> 1
        # x = [npix, infinity] -> 0
        # linearly between those, likewise for y.
        out = (
            self.psf["ll"] * wleft * wlow
            + self.psf["lr"] * (1 - wleft) * wlow
            + self.psf["ul"] * wleft * (1 - wlow)
            + self.psf["ur"] * (1 - wleft) * (1 - wlow)
        )
        return out

    def build_epsf_interpolator(
        self,
        image,
        oversamp_render=8,
        oversamp_taylor=50,
        order=1,
        max_radius=100,
        epsf=False,
    ):
        """Build the spatial Taylor expansions for an ePSF profile.

        Parameters
        ----------
        image : galsim.Image
            image within which we will inject PSFs
        oversamp_render : int
            Oversampling with which to render ePSFs using galsim.
            Should probably be an integer multiple of the native
            oversampling if using CRDS PSFs.
            Default 8
        oversamp_taylor : int
            Oversampling for the Taylor expansion.  Total RAM requirements
            will be ~oversamp_taylor**2*4*nterms*PSFstampsize times 4 bytes
            where nterms is 1, 3, or 6 depending on the desired order.
            For a 100x100 PSF, order=1, and oversampling of 50, this is
            about 1.2 GB.
            Default 50
        order : int
            Order of the Taylor expansion.  Must be 0, 1, or 2.  The number
            of terms in the Taylor expansion is 1, 3, or 6, respectively.
            Higher order = more RAM, more computational cost, but gives
            a higher accuracy at fixed oversamp_taylor.
            Default 1
        max_radius : int
            Maximum half-width of the box for the ePSF in pixels.  A very
            large box will be expensive in compute time and memory for
            the Taylor expansion.
            Default 100
        epsf : boolean
            Has the input PSF already been convolved with the pixel response
            function (is it an ePSF)?  If True, use no_pixel to render with
            galsim.
            Default False

        Returns
        -------
        None

        This routine builds the bounds and Taylor expansion arrays needed for
        draw_epsf; they are stored as self.bounds and self.psfinterpolators

        """

        if order not in [0, 1, 2]:
            raise ValueError(
                "Fast PSF interpolation only available for orders 0, 1, or 2."
            )

        # First, figure out how large to make the stamps.  Use the lower
        # left corner of the detector for this (we could use any spot).
        # Render one PSF and adopt the size of that stamp.

        pointsource = galsim.DeltaFunction()
        p = galsim.Convolve(pointsource, self.psf["ll"])

        image_pos = galsim.PositionD(
            self.corners["ll"][0], self.corners["ll"][1]
        )
        pwcs = image.wcs.local(image_pos)

        bounds = p.drawImage(center=(-0.5, -0.5), wcs=pwcs).bounds
        ncenter = max(-bounds.getXMin(), -bounds.getYMin())
        ncenter = min(ncenter, max_radius)

        bounds = galsim.BoundsI(-ncenter, ncenter, -ncenter, ncenter)

        dn = 2 * ncenter + 1

        self.bounds = bounds
        self.oversamp_taylor = oversamp_taylor
        self.order = order
        self.stampshape = (dn, dn)

        # These interpolate within the oversampled rendered PSFs.

        xinterp = np.arange(dn * oversamp_render) * 1.0 / oversamp_render
        yinterp = np.arange(dn * oversamp_render) * 1.0 / oversamp_render

        self.psfinterpolators = {
            "ll": None,
            "lr": None,
            "ul": None,
            "ur": None,
        }

        # Number of terms needed for a Taylor expansion of the desired order

        nterms = [1, 3, 6][order]

        shape = (oversamp_taylor + 1, oversamp_taylor + 1, 4, nterms, dn, dn)
        self.allarrays = np.zeros(shape, order="C", dtype=np.float32)

        # At each PSF location, render a PSF at many subpixel dithers.  Use
        # these together with bicubic interpolation to define the values and
        # derivatives on a very oversampled grid.

        for iloc, key in enumerate(["ll", "lr", "ul", "ur"]):
            image_pos = galsim.PositionD(
                self.corners[key][1], self.corners[key][0]
            )
            pwcs = image.wcs.local(image_pos)
            p = galsim.Convolve(pointsource, self.psf[key])

            # Render the PSF at subpixel positions using galsim

            nover = oversamp_render
            method = "no_pixel" if epsf else "auto"
            allrendered = np.zeros((dn * nover, dn * nover))

            for i in range(nover):
                for j in range(nover):
                    im = p.drawImage(
                        center=(-j / nover, -i / nover),
                        wcs=pwcs,
                        bounds=bounds,
                        method=method,
                    )
                    allrendered[i::nover, j::nover] = im.array

            f = interpolate.RectBivariateSpline(
                xinterp, yinterp, allrendered, kx=3, ky=3
            )
            fullarr_derivs = np.zeros(self.allarrays[:, :, 0].shape)

            # Save the Taylor expansion to an array

            for i in range(oversamp_taylor + 1):
                for j in range(oversamp_taylor + 1):
                    x = np.arange(dn) + i / oversamp_taylor
                    y = np.arange(dn) + j / oversamp_taylor
                    fullarr_derivs[i, j, 0] = f(x, y)
                    if order >= 1:
                        fullarr_derivs[i, j, 1] = f(x, y, dx=1)
                        fullarr_derivs[i, j, 2] = f(x, y, dy=1)
                    if order == 2:
                        fullarr_derivs[i, j, 3] = f(x, y, dx=2)
                        fullarr_derivs[i, j, 4] = f(x, y, dx=1, dy=1)
                        fullarr_derivs[i, j, 5] = f(x, y, dy=2)

            # Use this structure for an efficient memory layout,
            # since we will need the expansion at different PSF
            # locations at the same subpixel dither.

            self.allarrays[:, :, iloc] = fullarr_derivs
            self.psfinterpolators[key] = self.allarrays[:, :, iloc]

    def draw_epsf(self, x, y, fluxfactor=1):
        """Draw an ePSF at (x, y) using a Taylor expansion.

        Linearly interpolate between the four corners to obtain the
        PSF at this location.

        Parameters
        ----------
        x : float
            x position
        y : float
            y position
        fluxfactor : float
            factor by which to multiply the ePSF
            Default 1

        Returns
        -------
        GalSim image representing the ePSF at (x, y), including bounds.
        """

        npix = self.corners["ur"][-1]
        off = self.corners["ll"][0]

        wleft = np.clip((npix - x) / (npix - off), 0, 1)
        wlow = np.clip((npix - y) / (npix - off), 0, 1)

        # x = [0, off] -> 1
        # x = [npix, infinity] -> 0
        # linearly between those, likewise for y.

        # integer pixel:

        offset_x = int(np.ceil(x))
        offset_y = int(np.ceil(y))

        # Fractional part of a pixel we need to interpolate to

        dx = offset_x - x
        dy = offset_y - y

        # These are the integer and fractional parts of the fractional part
        # above, after converting to units of oversampled pixels.
        # Example: The Taylor expanion has oversampling of 10, and the
        # subpixel offset is 0.53.  The offset is 5 integer units plus
        # 0.3 fractional unit where the unit is oversampled pixels.

        int_x = int(self.oversamp_taylor * dx + 0.5)
        int_y = int(self.oversamp_taylor * dy + 0.5)
        frac_x = np.float32(dx - int_x / self.oversamp_taylor)
        frac_y = np.float32(dy - int_y / self.oversamp_taylor)

        # Blank stamp.  Keep everything in float32 for efficiency.

        epsf_out = np.zeros(self.stampshape, dtype=np.float32)

        weights = {
            "ll": wleft * wlow,
            "lr": (1 - wleft) * wlow,
            "ul": wleft * (1 - wlow),
            "ur": (1 - wleft) * (1 - wlow),
        }

        for key in ["ll", "lr", "ul", "ur"]:
            M = self.psfinterpolators[key]
            w = np.float32(weights[key] * fluxfactor)

            epsf_out += w * M[int_y, int_x, 0]

            if self.order >= 1:
                epsf_out += w * frac_y * M[int_y, int_x, 1]
                epsf_out += w * frac_x * M[int_y, int_x, 2]

            if self.order == 2:
                epsf_out += w / 2 * frac_y**2 * M[int_y, int_x, 3]
                epsf_out += w * frac_y * frac_x * M[int_y, int_x, 4]
                epsf_out += w / 2 * frac_x**2 * M[int_y, int_x, 5]

        stampbounds = self.bounds.shift(galsim.PositionI(offset_x, offset_y))

        return galsim.Image(epsf_out, bounds=stampbounds)


@cache
def get_epsf_from_crds(sca, filter_name, date=None):
    """Retrieve EPSF reference model from CRDS

    Parameters
    ----------
    sca : int
        SCA number
    filter_name : str
        name of filter
    date : astropy.time.Time or None
        Date of simulation. If None, the default from the parameters configuration `default_date` is used

    Returns
    -------
    model : roman_datamodels.EpsfRefModel
    """
    from crds import getreferences

    if date is None:
        date = default_date
        log.warning(
            "No date has been specified for CRDS EPSF retrieval. Using %s",
            date.isot,
        )
    header = {
        "ROMAN.META.INSTRUMENT.NAME": "wfi",
        "ROMAN.META.INSTRUMENT.DETECTOR": f"SCA{sca:02d}",
        "ROMAN.META.INSTRUMENT.OPTICAL_ELEMENT": galsim2roman_bandpass[
            filter_name
        ],
        "ROMAN.META.EXPOSURE.START_TIME": date.isot,
    }
    ref_paths = getreferences(header, reftypes=["epsf"], observatory="roman")
    model = datamodels.open(ref_paths["epsf"])

    return model


@cache
def get_gridded_psf_model(
    psf_ref_model, oversample=None, focus=0, spectral_type=1
):
    """Generate the gridded PSF model from an EPSF reference model

    Compute a gridded PSF model for one SCA using the
    reference files in CRDS.
    The input reference files have 3 focus positions and this is using
    the in-focus images. There are also three spectral types that are
    available and this code uses the M5V spectal type.
    """
    # Open the reference file data model
    # select the infocus images (0) and we have a selection of spectral types
    # A0V, G2V, and M6V, pick G2V (1)
    psf_images = psf_ref_model.psf[focus, spectral_type, :, :, :].copy()

    # get the central position of the cutouts in a list
    psf_positions_x = psf_ref_model.meta.pixel_x.data.data
    psf_positions_y = psf_ref_model.meta.pixel_y.data.data
    meta = OrderedDict()

    # Create the GriddedPSFModel
    position_list = []
    for index in range(len(psf_positions_x)):
        position_list.append([psf_positions_x[index], psf_positions_y[index]])
    meta["grid_xypos"] = position_list
    if oversample is None:
        oversample = psf_ref_model.meta.oversample
    meta["oversampling"] = oversample
    meta["epsf_oversample"] = psf_ref_model.meta.oversample
    nd = NDData(psf_images, meta=meta)
    model = GriddedPSFModel(nd)

    return model


def make_one_psf(
    sca,
    filter_name,
    wcs=None,
    psftype="galsim",
    pix=None,
    chromatic=False,
    oversample=4,
    extra_convolution=None,
    date=None,
    **kw,
):
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
    psftype : One of ['epsf', 'galsim', 'stpsf']
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
        Date of simulation. If None, current date is used. Needed for psftype='epsf'
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
    pix = pix if pix is not None else (n_pix // 2, n_pix // 2)
    if wcs is None:
        log.warning("wcs is None; unlikely to get orientation of PSF correct.")

    # Create the PSF depending on method desired.
    if psftype == "stpsf":
        psf = make_one_psf_stpsf(
            sca,
            filter_name,
            wcs=wcs,
            pix=pix,
            chromatic=chromatic,
            oversample=oversample,
            extra_convolution=extra_convolution,
            **kw,
        )
    elif psftype == "epsf":
        psf = make_one_psf_epsf(
            sca,
            filter_name,
            wcs=wcs,
            pix=pix,
            chromatic=chromatic,
            extra_convolution=extra_convolution,
            date=date,
            **kw,
        )
    else:  # Default is galsim
        psf = make_one_psf_galsim(
            sca,
            filter_name,
            wcs=wcs,
            pix=pix,
            chromatic=chromatic,
            extra_convolution=extra_convolution,
            **kw,
        )

    return psf


def make_one_psf_epsf(
    sca,
    filter_name,
    wcs=None,
    pix=None,
    chromatic=False,
    extra_convolution=None,
    date=None,
    **kw,
):
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
        Date of simulation. If None, current date is used. Needed for psftype='epsf'
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
    log.info("Creating PSF from CRDS reference type epsf")
    if chromatic:
        log.warning(
            "romanisim does not yet support chromatic PSFs with stpsf or crds epsf"
        )
    epsf_ref_model = get_epsf_from_crds(sca, filter_name, date=date)
    gridded_psf = get_gridded_psf_model(epsf_ref_model)

    psf = psf_from_grid(gridded_psf, *pix)
    pixelscale = pixel_scale / gridded_psf.meta["epsf_oversample"]
    intimg = psfstamp_to_galsimimage(
        psf, pixelscale, wcs=wcs, pix=pix, extra_convolution=extra_convolution
    )
    return intimg


def make_one_psf_galsim(
    sca,
    filter_name,
    wcs=None,
    pix=None,
    chromatic=False,
    extra_convolution=None,
    **kw,
):
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
    log.info("Creating PSF using galsim")
    filter_name = roman2galsim_bandpass[filter_name]
    defaultkw = {"pupil_bin": 8}
    if chromatic:
        defaultkw["n_waves"] = 10
        bandpass = None
    else:
        bandpass = getBandpasses(AB_zeropoint=True)[filter_name]
        filter_name = None
    defaultkw.update(**kw)
    scapos = galsim.PositionD(*pix) if pix is not None else None
    res = getPSF(
        sca,
        filter_name,
        wcs=wcs,
        SCA_pos=scapos,
        wavelength=bandpass,
        **defaultkw,
    )
    if extra_convolution is not None:
        res = galsim.Convolve(res, extra_convolution)
    return res


def make_one_psf_stpsf(
    sca,
    filter_name,
    wcs=None,
    pix=None,
    chromatic=False,
    oversample=4,
    extra_convolution=None,
    **kw,
):
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
        depending on whether stpsf is set. May also include "stpsf_options"
        dictionary to specify WFI object options (e.g. defocus, jitter)

    Returns
    -------
    profile : galsim.gsobject.GSObject
        galsim profile object for convolution with source profiles when
        rendering scenes.
    """
    log.info("Creating PSF using stpsf")
    if chromatic:
        log.warning("romanisim does not yet support chromatic PSFs with stpsf")

    import stpsf as wpsf

    filter_name = galsim2roman_bandpass[filter_name]
    wfi = wpsf.WFI()
    wfi.detector = f"SCA{sca:02d}"
    wfi.filter = filter_name
    wfi.detector_position = pix

    # Extract STPSF object options and function arguments separately
    opts = kw.pop("stpsf_options", {})
    args = kw
    for key, value in opts.items():
        wfi.options[key] = value

    psf = wfi.calc_psf(oversample=oversample, **args)
    pixelscale = wfi.pixelscale / oversample
    intimg = psfstamp_to_galsimimange(
        psf[0].data,
        pixelscale,
        wcs=wcs,
        pix=pix,
        extra_convolution=extra_convolution,
    )
    return intimg


def make_psf(
    sca,
    filter_name,
    wcs=None,
    psftype="galsim",
    pix=None,
    chromatic=False,
    variable=False,
    extra_convolution=None,
    date=None,
    **kw,
):
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
    psftype : One of ['epsf', 'galsim', 'stpsf]
        How to determine the PSF.
    pix : tuple (float, float)
        pixel location of PSF on focal plane
    variable : bool
        True if a variable PSF object is desired
    date : astropy.time.Time or None
        Date of simulation. If None, current date is used. Needed for psftype='epsf'
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
        return make_one_psf(
            sca,
            filter_name,
            wcs=wcs,
            psftype=psftype,
            pix=pix,
            chromatic=chromatic,
            extra_convolution=extra_convolution,
            date=date,
            **kw,
        )
    elif pix is not None:
        raise ValueError("cannot set both pix and variable")
    buf = 49
    # Stpsf complains if we get too close to (0, 0) for some reason.
    # For other corners one can go to within a fraction of a pixel.
    # if we go larger than 49 we have to change some of the tests, which use a 100x100 image.
    corners = dict(
        ll=[buf, buf],
        lr=[n_pix - buf, buf],
        ul=[buf, n_pix - buf],
        ur=[n_pix - buf, n_pix - buf],
    )
    psfs = dict()
    for corner, pix in corners.items():
        psfs[corner] = make_one_psf(
            sca,
            filter_name,
            wcs=wcs,
            psftype=psftype,
            pix=pix,
            chromatic=chromatic,
            extra_convolution=extra_convolution,
            **kw,
        )
    return VariablePSF(corners, psfs)


def psf_from_grid(psfgrid, x_0=None, y_0=None, size=185):
    """Calculate a PSF profile from a GriddedPSFModel at the specified position

    Parameters
    ----------
    psfgrid : GriddedPSFModel
        The PSF model to calculate from

    x_0, y_0 : float or None
        Position to calculate the psf. If None, (0., 0.) is used

    size : int
        Stamp size. Must be odd.
        The default, 185, is the default stamp size for the STPSF stamp.

    Returns
    -------
    psf : nd.array
        The psf profile.
    """
    if size % 2 == 0:
        raise ValueError(
            f"Argument `size` is required to be odd. Given: {size}"
        )

    x_0 = 2048 if x_0 is None else x_0
    y_0 = 2048 if y_0 is None else y_0
    cc = (np.arange(size) - (size // 2)) / psfgrid.meta["epsf_oversample"]
    x, y = np.meshgrid(cc + x_0, cc + y_0)
    psf = psfgrid.evaluate(x, y, 1, x_0, y_0)
    return psf


def psfstamp_to_galsimimage(
    psf, pixelscale, wcs=None, pix=None, extra_convolution=None
):
    """Convert an STPSF/CRDS PSF profile to galsim.Image"""

    # stpsf doesn't do distortion
    # calc_psf gives something aligned with the pixels, but with
    # a constant pixel scale equal to wfi.pixelscale / oversample.
    # we need to get the appropriate rotated WCS that matches this
    if wcs is not None:
        local_jacobian = wcs.local(image_pos=galsim.PositionD(pix)).getMatrix()
        # angle of [du/dx, du/dy]
        ang = np.arctan2(local_jacobian[0, 1], local_jacobian[0, 0])
        rotmat = np.array(
            [[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]
        )
        newwcs = galsim.JacobianWCS(*(rotmat.ravel() * pixelscale))
        # we are making a new, orthogonal, isotropic matrix for the PSF with the
        # appropriate pixel scale.  This is intended to be the WCS for the PSF
        # produced by stpsf.
    else:
        newwcs = galsim.JacobianWCS(*(np.array([1, 0, 0, 1]) * pixelscale))
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
        gimg, normalization="flux", use_true_center=True, offset=centroid
    )

    if extra_convolution is not None:
        intimg = galsim.Convolve(intimg, extra_convolution)

    return intimg


def psfstamp_to_galsimimange(
    psf, pixelscale, wcs=None, pix=None, extra_convolution=None
):
    """Convert an STPSF/CRDS PSF profile to galsim.Image"""

    # stpsf doesn't do distortion
    # calc_psf gives something aligned with the pixels, but with
    # a constant pixel scale equal to wfi.pixelscale / oversample.
    # we need to get the appropriate rotated WCS that matches this
    if wcs is not None:
        local_jacobian = wcs.local(image_pos=galsim.PositionD(pix)).getMatrix()
        # angle of [du/dx, du/dy]
        ang = np.arctan2(local_jacobian[0, 1], local_jacobian[0, 0])
        rotmat = np.array(
            [[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]
        )
        newwcs = galsim.JacobianWCS(*(rotmat.ravel() * pixelscale))
        # we are making a new, orthogonal, isotropic matrix for the PSF with the
        # appropriate pixel scale.  This is intended to be the WCS for the PSF
        # produced by stpsf.
    else:
        newwcs = galsim.JacobianWCS(*(np.array([1, 0, 0, 1]) * pixelscale))
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
        gimg, normalization="flux", use_true_center=True, offset=centroid
    )

    if extra_convolution is not None:
        intimg = galsim.Convolve(intimg, extra_convolution)

    return intimg
