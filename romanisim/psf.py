from collections import OrderedDict
from functools import cache

import galsim
import numpy as np

from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel
from roman_datamodels import datamodels
from scipy import interpolate

from romanisim import log
from romanisim.models import ipc

from .models.bandpass import getBandpasses, galsim2roman_bandpass, roman2galsim_bandpass
from .models.parameters import (
    default_date,
    reference_data,
    n_pix,
    pixel_scale,
)
from .models.psf_utils import getPSF

__all__ =  ['VariablePSF',
            'deconvolve_ipc',
            'psf_stamp_wcs',
            'epsf_is_pixel_convolved',
            'get_epsf_from_crds',
            'get_gridded_psf_model',
            'make_one_psf',
            'make_one_psf_epsf',
            'make_one_psf_galsim',
            'make_one_psf_stpsf',
            'make_psf',
            'psf_from_grid',
            'psfstamp_to_galsimimage',]


class VariablePSF:
    """Spatially variable PSF wrapping GalSim profiles.

    Linearly interpolates between four corner PSF profiles by summing
    weighted GalSim PSF profiles.
    """

    def __init__(self, corners, psf):
        self.corners = corners
        self.psf = psf
        self.psfinterpolators = None
        # True if the corner profiles already include the pixel response
        # function of the grid they will be drawn onto.
        self.pixel_convolved = all(
            getattr(p, "pixel_convolved", False) for p in psf.values())

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
                self.corners[key][0], self.corners[key][1]
            )
            pwcs = image.wcs.local(image_pos)
            p = galsim.Convolve(pointsource, self.psf[key])

            # Render the PSF at subpixel positions using galsim

            nover = oversamp_render
            method = "no_pixel" if self.pixel_convolved else "auto"
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

    override = reference_data.get('epsf')
    if isinstance(override, str):
        log.info('Forcing the use of ePSF reference %s.', override)
        return datamodels.open(override)

    if date is None:
        date = default_date
        log.warning(
            "No date has been specified for CRDS EPSF retrieval. Using %s",
            date.isot,
        )
    header = {
        "ROMAN.META.INSTRUMENT.NAME": "wfi",
        "ROMAN.META.INSTRUMENT.DETECTOR": f"WFI{sca:02d}",
        "ROMAN.META.INSTRUMENT.OPTICAL_ELEMENT": galsim2roman_bandpass[
            filter_name
        ],
        "ROMAN.META.EXPOSURE.START_TIME": date.isot,
    }
    ref_paths = getreferences(header, reftypes=["epsf"], observatory="roman")
    model = datamodels.open(ref_paths["epsf"])

    return model


def epsf_is_pixel_convolved(psf_ref_model, focus=0, spectral_type=1):
    """Determine if an epsf reference file has been pixel-convolved.

    We would like a metadata flag saying whether the PSF has been convolved
    with the pixel response function, but the reference files do not carry
    one, so we key off the normalization instead.  romancal makes the same
    determination the same way.

    The convention for a pixel-convolved reference is that taking every
    ``oversample``-th sample gives exactly the fraction of the flux that
    would land in real native pixels at that subpixel offset.  This convention
    leads to an overall normalization of about ``oversample ** 2``.  An
    older, un-convolved reference is normalized to the enclosed-flux
    fraction and sums to slightly less than one.  We cut at a sum of 1.1.

    A reference file following either convention should land close to one
    of those two values; we warn if it does not, since that suggests the
    file follows some third convention that we are guessing about.

    Parameters
    ----------
    psf_ref_model : roman_datamodels.EpsfRefModel
        The reference model to inspect.
    focus, spectral_type : int
        Indices of the plane to test; any plane will do.

    Returns
    -------
    bool
        True if the reference PSF has been convolved with the pixel
        response function.
    """
    total = np.sum(psf_ref_model.psf[focus, spectral_type, 0, :, :])
    pixel_convolved = bool(total > 1.1)  # a bit more than 1, for buffer
    expected = psf_ref_model.meta.oversample ** 2 if pixel_convolved else 1
    if not (0.9 * expected < total < 1.05 * expected):
        log.warning(
            'EPSF reference sums to %f, which is not close to the expected '
            '%d; is this reference file following a different convention?',
            total, expected)
    return pixel_convolved


def deconvolve_ipc(psf_images, ipc_kernel, oversample, pad=32):
    """Remove interpixel capacitance from oversampled PSF stamps.

    IPC couples whole native pixels, so on a stamp oversampled by
    ``oversample`` it acts as a convolution with a sparse kernel
    which is non-zero only at pixels spaced
    ``oversample`` samples apart.  That comb has transfer function

        A(f) = sum_ij a_ij exp(-2 pi i oversample (i f_y + j f_x))

    which we divide out, being careful to leave the PSF centering
    unaffected.

    romanisim deconvolves using this function and reconvolves later in
    ``romanisim.l1.make_l1`` with the same kernel.  Away from
    the stamp edges the two cancel to about 1e-16 of the PSF peak.

    Parameters
    ----------
    psf_images : np.ndarray[n_psf, ny, nx]
        Oversampled PSF stamps, IPC included.
    ipc_kernel : np.ndarray[n, n]
        The IPC kernel; n must be odd.  Normalized here if it is not
        already.
    oversample : int
        Extent to which the PSF stamp was oversampled
    pad : int
        Zero padding added before the FFT so that flux does not wrap
        around the edge of the stamp.

    Returns
    -------
    np.ndarray[n_psf, ny, nx]
        The stamps with IPC removed.
    """
    ipc_kernel = np.asarray(ipc_kernel, dtype=float)
    if ipc_kernel.ndim != 2 or ipc_kernel.shape[0] != ipc_kernel.shape[1]:
        raise ValueError('IPC kernel must be square and two dimensional')
    if ipc_kernel.shape[0] % 2 == 0:
        raise ValueError('IPC kernel must have a center; its size must be odd')
    ipc_kernel = ipc_kernel / np.sum(ipc_kernel)
    ny, nx = psf_images.shape[-2:]
    padded = np.pad(np.asarray(psf_images, dtype=float),
                    ((0, 0), (pad, pad), (pad, pad)))

    fy = np.fft.fftfreq(padded.shape[-2])[:, None]
    fx = np.fft.rfftfreq(padded.shape[-1])[None, :]
    transfer = np.zeros((fy.size, fx.size), dtype=complex)
    nkern = ipc_kernel.shape[0] // 2
    for i, dy in enumerate(range(-nkern, nkern + 1)):
        for j, dx in enumerate(range(-nkern, nkern + 1)):
            transfer += ipc_kernel[i, j] * np.exp(
                -2j * np.pi * oversample * (dy * fy + dx * fx))

    out = np.fft.irfft2(np.fft.rfft2(padded) / transfer,
                        s=padded.shape[-2:])
    return out[:, pad:pad + ny, pad:pad + nx]


def get_gridded_psf_model(
    psf_ref_model, oversample=None, focus=0, spectral_type=1,
    ipc_kernel=None,
):
    """Generate the gridded PSF model from an EPSF reference model

    Compute a gridded PSF model for one SCA using the
    reference files in CRDS.
    The input reference files have 3 focus positions and this is using
    the in-focus images. There are also three spectral types that are
    available and this code uses the M5V spectal type.

    Two conventions for the reference file are supported; see
    `epsf_is_pixel_convolved` for how they are distinguished.
    Older reference files store the optical PSF without convolution
    by the pixel response function, not including distortion, and
    containing a 'psf_noipc' extension that does not include the effect
    of IPC.  Newer reference files include the pixel response function,
    distortion, and IPC (i.e., are what an empirical view of the PSF
    would look like on real data).

    Parameters
    ----------
    ipc_kernel : np.ndarray[n, n] or None
        The IPC kernel that ``make_l1`` will apply.  Only used for
        pixel-convolved references.  If None,
        ``romanisim.models.ipc.ipc_kernel`` is used, which is also what
        ``make_l1`` falls back to.

    Returns
    -------
    photutils.psf.GriddedPSFModel
        The gridded model.  ``model.meta["pixel_convolved"]`` records which
        convention the reference file followed.
    """
    # Open the reference file data model
    # select the infocus images (0) and we have a selection of spectral types
    # A0V, G2V, and M6V, pick G2V (1)
    oversample_ref = psf_ref_model.meta.oversample
    pixel_convolved = epsf_is_pixel_convolved(psf_ref_model, focus=focus,
                                              spectral_type=spectral_type)

    if pixel_convolved:
        psf_images = psf_ref_model.psf[focus, spectral_type, :, :, :].copy()
        # Each sample is the flux that would land in a whole native pixel
        # centered there; galsim wants the flux in one oversampled sample.
        psf_images = psf_images / oversample_ref ** 2
        if ipc_kernel is None:
            ipc_kernel = ipc.ipc_kernel
        psf_images = deconvolve_ipc(psf_images, ipc_kernel, oversample_ref)
    else:
        psf_images = psf_ref_model.psf_noipc[
            focus, spectral_type, :, :, :].copy()

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
    meta["epsf_oversample"] = oversample_ref
    meta["pixel_convolved"] = pixel_convolved
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
    ipc_kernel=None,
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
            ipc_kernel=ipc_kernel,
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
    ipc_kernel=None,
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
    ipc_kernel : np.ndarray[3, 3] or None
        The IPC kernel to deconvolve from the ePSF.  It should match
        the kernel that is later reapplied in romanisim.l1.make_l1.
        If None,
        romanisim.models.ipc.ipc_kernel is used, which is also what make_l1
        falls back to.
    **kw : dict
        Additional keywords passed to galsim.roman.getPSF or stpsf.calc_psf,
        depending on whether stpsf is set.

    Returns
    -------
    profile : galsim.gsobject.GSObject
        galsim profile object for convolution with source profiles when
        rendering scenes.  The ``pixel_convolved`` attribute records whether
        the profile already includes the pixel response function; see
        `romanisim.image.add_objects_to_image`.
    """
    log.info("Creating PSF from CRDS reference type epsf")
    if chromatic:
        log.warning(
            "romanisim does not yet support chromatic PSFs with stpsf or crds epsf"
        )
    epsf_ref_model = get_epsf_from_crds(sca, filter_name, date=date)
    gridded_psf = get_gridded_psf_model(epsf_ref_model, ipc_kernel=ipc_kernel)

    psf = psf_from_grid(gridded_psf, *pix)
    pixel_convolved = gridded_psf.meta["pixel_convolved"]
    oversample = gridded_psf.meta["epsf_oversample"]
    if pixel_convolved:
        stampwcs = psf_stamp_wcs(wcs=wcs, pix=pix, oversample=oversample)
    else:
        stampwcs = psf_stamp_wcs(wcs=wcs, pix=pix,
                                 samplescale=pixel_scale / oversample)
    intimg = psfstamp_to_galsimimage(
        psf, stampwcs, extra_convolution=extra_convolution)
    intimg.pixel_convolved = pixel_convolved
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
    # STPSF exposes the grism as two diffraction orders, GRISM0 (0th) and
    # GRISM1 (1st).  Default to the 1st order, which is what callers asking
    # for --bandpass GRISM almost always want.
    if filter_name == "GRISM":
        wfi.filter = "GRISM1"
    else:
        wfi.filter = filter_name
    wfi.detector_position = pix

    # Extract STPSF object options and function arguments separately
    opts = kw.pop("stpsf_options", {})
    args = kw
    for key, value in opts.items():
        wfi.options[key] = value

    psf = wfi.calc_psf(oversample=oversample, **args)
    # stpsf does not apply distortion; calc_psf gives something aligned with
    # the pixels, but with a constant sample scale.
    stampwcs = psf_stamp_wcs(wcs=wcs, pix=pix,
                             samplescale=wfi.pixelscale / oversample)
    intimg = psfstamp_to_galsimimage(
        psf[0].data, stampwcs, extra_convolution=extra_convolution)
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
    ipc_kernel=None,
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
    ipc_kernel : np.ndarray[3, 3] or None
        The IPC kernel that romanisim.l1.make_l1 will apply; only used for
        psftype='epsf'.  See make_one_psf_epsf.
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
            ipc_kernel=ipc_kernel,
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
            date=date,
            ipc_kernel=ipc_kernel,
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


def psf_stamp_wcs(wcs=None, pix=None, samplescale=None, oversample=None):
    """Build the WCS for an oversampled PSF stamp.

    A PSF can come from one of two sources, controlled by oversample or samplescale.

    Give ``samplescale`` for a stamp on an idealized grid of square samples
    of that angular size (e.g., from stpsf).  This WCS will include the
    rotation of the image relative to north and the given pixel scale.

    Give ``oversample`` for a stamp that samples the native detector grid.
    Its axes are the detector axes, so the local Jacobian maps them to the
    sky directly and the PSF WCS needs the local plate scale and shear.

    Parameters
    ----------
    wcs : callable or None
        WCS of the image the PSF will be rendered into.  If None, a default
        North-up WCS is used and ``samplescale`` is required.
    pix : tuple (float, float)
        Pixel location of the PSF on the focal plane.
    samplescale : float or None
        Angular size of one sample of the stamp, in arcseconds.
    oversample : int or None
        Samples per native pixel.

    Returns
    -------
    galsim.JacobianWCS
    """
    if (samplescale is None) == (oversample is None):
        raise ValueError('give exactly one of samplescale and oversample')

    if wcs is None:
        if oversample is not None:
            raise ValueError('a native-pixel stamp needs a wcs to get the '
                             'local Jacobian from')
        # just use a default North = up WCS
        return galsim.JacobianWCS(*(np.array([1, 0, 0, 1]) * samplescale))

    jacobian = wcs.local(image_pos=galsim.PositionD(pix)).getMatrix()
    if oversample is not None:
        return galsim.JacobianWCS(*(jacobian.ravel() / oversample))

    # An idealized stamp needs only the orientation of the local pixels.  We make
    # a new orthogonal, isotropic matrix for the PSF with the appropriate sample
    # scale; the angle is that of [du/dx, du/dy].
    ang = np.arctan2(jacobian[0, 1], jacobian[0, 0])
    rotmat = np.array([[np.cos(ang), np.sin(ang)],
                       [-np.sin(ang), np.cos(ang)]])
    return galsim.JacobianWCS(*(rotmat.ravel() * samplescale))


def psfstamp_to_galsimimage(psf, stampwcs, extra_convolution=None):
    """Convert an STPSF/CRDS PSF stamp to a galsim profile.

    Parameters
    ----------
    psf : np.ndarray
        The oversampled PSF stamp.
    stampwcs : galsim.wcs.BaseWCS
        WCS of the stamp; see `psf_stamp_wcs`.
    extra_convolution : galsim.gsobject.GSObject or None
        Additional convolution to add to the PSF.

    Returns
    -------
    galsim.InterpolatedImage
    """
    gimg = galsim.Image(psf, wcs=stampwcs)

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
