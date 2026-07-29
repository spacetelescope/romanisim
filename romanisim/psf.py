from collections import OrderedDict
from functools import cache

import galsim
import numpy as np

from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel
from roman_datamodels import datamodels
from scipy import interpolate, signal

from romanisim import log

from romanisim.models import ipc

from .models.bandpass import getBandpasses, galsim2roman_bandpass, roman2galsim_bandpass
from .models.parameters import (
    default_date,
    n_pix,
    pixel_scale,
)
from .models.psf_utils import getPSF

__all__ =  ['VariablePSF',
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

    def __init__(self, corners, psf, psftype):
        self.corners = corners
        self.psf = psf
        self.psfinterpolators = None
        self.psftype = psftype

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
                self.corners[key][1], self.corners[key][0]
            )
            pwcs = image.wcs.local(image_pos)
            p = galsim.Convolve(pointsource, self.psf[key])

            # Render the PSF at subpixel positions using galsim

            nover = oversamp_render
            method = "no_pixel" if self.psftype == "epsf" else "auto"
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
        psf_ref_model,
        oversample=None,
        focus=0,
        spectral_type=1,
        extra_kernel=None
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

    if extra_kernel is not None:
        psf_images = signal.convolve(psf_images,
                                     extra_kernel[None, :, :],
                                     mode='same',
                                     method='direct'
                                     )

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
    ipc_kernel : ndarray or None
        Kernel, 3x3 or 5x5, representing the interpixel capacitance (IPC)
        kernel.  If psftype is 'epsf' then the epsf will be deconvolved
        with the IPC kernel to compensate for the IPC convolution that is
        applied later in l1.py.  If ipc_kernel is None, load the kernel
        from the ipc module.
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
    ipc_kernel : ndarray or None
        Kernel, 3x3 or 5x5, representing the interpixel capacitance (IPC)
        kernel.  The epsf will be deconvolved with the IPC kernel to
        compensate for the IPC convolution that is applied later in l1.py.
        If ipc_kernel is None, load the kernel from the ipc module.
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

    # Deconvolve with IPC.  First get the IPC kernel, then derive the
    # corresponding deconvolution kernel, then apply it.  The IPC kernel
    # is small, typically 3x3, so we need some zero padding to compute a
    # good deconvolution kernel using Fourier methods.  Then trim the
    # deconvolution kernel to 5x5.

    if ipc_kernel is None:
        ipc_kernel = ipc.ipc_kernel

    padded_kernel = np.pad(ipc_kernel, 20)

    deltafunc = np.zeros(padded_kernel.shape)
    deltafunc[padded_kernel.shape[0]//2, padded_kernel.shape[1]//2] = 1

    deconvolution_kernel = create_convolution_kernel(padded_kernel, deltafunc, size=5)

    # Now we need to embed the deconvolution kernel sparsely within a
    # larger array with the appropriate oversampling.

    oversample = gridded_psf.meta["epsf_oversample"]
    oversampled_kernel = np.zeros((4*oversample + 1, 4*oversample + 1))
    oversampled_kernel[::oversample, ::oversample] = deconvolution_kernel

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
    pixelscale = wfi.pixelscale / oversample
    intimg = psfstamp_to_galsimimage(
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
    extra_convolution : galsim.gsobject.GSObject or None
        Additional convolution to add to PSF profiles
    date : astropy.time.Time or None
        Date of simulation. If None, current date is used. Needed for psftype='epsf'
        to choose the appropriate epsf reference.
    ipc_kernel : ndarray or None
        Kernel, 3x3 or 5x5, representing the interpixel capacitance (IPC)
        kernel.  If psftype is 'epsf' then the epsf will be deconvolved
        with the IPC kernel to compensate for the IPC convolution that is
        applied later in l1.py.  If ipc_kernel is None, load the kernel
        from the ipc module.
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
            ipc_kernel=ipc_kernel,
            **kw,
        )
    return VariablePSF(corners, psfs, psftype)


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


def create_convolution_kernel(
    input_psf, target_psf, min_fft_power_ratio=1e-5, size=None
):
    """Find convolution kernel which convolves input_psf to match target_psf.

    The nominal photutils matching kernel code does a straight ratio
    of the target and input PSFs in Fourier space.  This is a little
    fraught for our PSFs where they go to ~0 at high frequencies, and
    unless the window function is also exactly zero there, you can get
    huge amounts of power.

    We mitigate this by taking an approach analogous to Boucaud+2016,
    adding a regularizing term to the denominator of the FFT.  The
    scale of this term is set by min_fft_power_ratio; its amplitude
    will be the maximum of the power in the input PSF times this
    ratio.  Large values correspond to stronger regularization.

    Boucaud+2016 penalizes variation in the matching kernel, while the
    approach taken here simply penalizes large values in the matching
    kernel.  Other approaches like penalizing second or third
    derivatives of the matching kernel could be taken, but for the
    light regularization needed here the amplitude of the matching
    kernel was adequate.

    Parameters
    ----------
    input_psf : np.ndarray
        The input PSF which needs to be convolved

    target_psf : np.ndarray
        The target PSF which input_psf should match following convolution

    min_fft_power_ratio : float
        controls the scale of the regularization of the matching kernel in
        terms of the peak power of the input PSF's FFT.

    size : int
        the desired size of the final stamp

    Returns
    -------
    kernel : np.ndarray
        Convolution kernel taking input_psf to target_psf

    """
    input_psf = np.fft.ifftshift(input_psf.copy())
    target_psf = np.fft.ifftshift(target_psf.copy())

    input_psf /= input_psf.sum()
    target_psf /= target_psf.sum()

    input_fft = np.fft.fft2(input_psf)
    target_fft = np.fft.fft2(target_psf)

    input_power = np.abs(input_fft) ** 2
    max_power = np.max(input_power)
    conv_kernel_fft = (
        target_fft
        * np.conj(input_fft)
        / (input_power + min_fft_power_ratio * max_power)
    )

    kernel = np.real(np.fft.fftshift(np.fft.ifft2(conv_kernel_fft)))
    kernel = kernel / kernel.sum()
    
    if size is not None:
        return central_stamp(kernel, size).copy()
    return kernel


def central_stamp(im, size):
    """Extract the central region of an image.

    The image must be square and we extract a square stamp.  The parity
    of the size of im and size must be the same; if they are not, we add
    one to size so that its parity matches that of the size of im.  The
    motivation for this behavior is that it's not clear, for example,
    what it means to extract the 'central' 1x1 pixel region from a 2x2
    stamp.

    Parameters
    ----------
    im : np.ndarray
        the image
    size : int
        the number of pixels to extract

    Returns
    -------
    stamp : np.ndarray[size, size]
        the central pixels of im, possibly adjusted by 1 to account
        for parity differences
    """

    if im.shape[0] != im.shape[1]:
        raise ValueError("im must be square")
    if (im.shape[0] % 2) != (size % 2):
        size = size + 1
    parity = int((im.shape[0] % 2) == 0)
    center = im.shape[0] // 2
    sizeo2 = size // 2
    return im[
        center - sizeo2 : center + sizeo2 + 1 - parity,
        center - sizeo2 : center + sizeo2 + 1 - parity,
    ]
