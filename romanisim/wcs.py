"""Roman WCS interface for galsim.

galsim.roman has an implementation of Roman's WCS based on some SIP
coefficients for each SCA.  This could presumably be adequate, but here
we take the alternative approach of using the distortion functions
provided in CRDS.  These naturally are handled by the gWCS library,
but galsim only naturally supports astropy WCSes via the ~legacy
interface.  So this module primarily makes the wrapper that interfaces
gWCS and galsim.CelestialWCS together.

This presently gives rather different world coordinates given a specific
telescope boresight.  Partially this is not doing the same roll_ref
determination that galsim.roman does, which could be fixed.  But additionally
the center of the SCA looks to be in a different place relative to the
boresight for galsim.roman than for what I get from CRDS.  This bears
more investigation.
"""

import os
import warnings
import math
import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy.modeling import models
import astropy.time
import roman_datamodels
import gwcs.geometry
from gwcs import coordinate_frames as cf
import gwcs.wcs
import galsim.wcs
from galsim import roman
from . import util
import romanisim.parameters
import roman_datamodels.maker_utils as maker_utils


def fill_in_parameters(parameters, coord, pa_aper=0, boresight=True):
    """Add WCS info to parameters dictionary.

    Parameters
    ----------
    parameters : dict
        Metadata dictionary.  Dictionaries like pointing, aperture, and
        wcsinfo may be modified

    coord : astropy.coordinates.SkyCoord or galsim.CelestialCoord
        world coordinates at V2 / V3 ref (boresight or center of WFI SCAs)

    pa_aper : float
        position angle (North relative to the idealized Y axis) at the aperture
        V2Ref/V3Ref

    boresight : bool
        whether coord is the telescope boresight (V2 = V3 = 0) or the center of
        the WFI SCA array
    """
    coord = util.skycoord(coord)

    if 'pointing' not in parameters.keys():
        parameters['pointing'] = {}
    if 'wcsinfo' not in parameters.keys():
        parameters['wcsinfo'] = {}

    parameters['pointing']['target_ra'] = coord.ra.to(u.deg).value
    parameters['wcsinfo']['ra_ref'] = parameters['pointing']['target_ra']

    parameters['pointing']['target_dec'] = coord.dec.to(u.deg).value
    parameters['wcsinfo']['dec_ref'] = parameters['pointing']['target_dec']

    # Romanisim uses ROLL_REF = PA_APER - V3IdlYAngle
    parameters['wcsinfo']['roll_ref'] = (
        pa_aper - romanisim.parameters.V3IdlYAngle)

    if boresight:
        parameters['wcsinfo']['v2_ref'] = 0
        parameters['wcsinfo']['v3_ref'] = 0
        parameters['wcsinfo']['aperture_name'] = 'BORESIGHT'
        parameters['pointing']['target_aperture'] = 'BORESIGHT'
    else:
        from .parameters import v2v3_wficen
        parameters['wcsinfo']['v2_ref'] = v2v3_wficen[0]
        parameters['wcsinfo']['v3_ref'] = v2v3_wficen[1]
        parameters['wcsinfo']['aperture_name'] = 'WFI_CEN'
        parameters['pointing']['target_aperture'] = 'WFI_CEN'

    parameters['wcsinfo']['pa_aperture'] = pa_aper


def get_wcs(image, usecrds=True, distortion=None):
    """Get a WCS object for a given SCA or set of CRDS parameters.

    Parameters
    ----------
    image : roman_datamodels.datamodels.ImageModel or dict
        Image model or dictionary containing CRDS parameters
        specifying appropriate reference distortion
        map to load.
    usecrds : bool
        If True, use crds reference distortions rather than galsim.roman
        distortion model.
    distortion : astropy.modeling.core.CompoundModel
        Coordinate distortion transformation parameters

    Returns
    -------
    galsim.CelestialWCS for an SCA
    """

    # If sent a dictionary, create a temporary model for CRDS interface
    if (type(image) is not roman_datamodels.datamodels.ImageModel):
        image_node = maker_utils.mk_level2_image()
        for key in image.keys():
            if isinstance(image[key], dict):
                image_node['meta'][key].update(image[key])
            else:
                image_node['meta'][key] = image[key]
        image_mod = roman_datamodels.datamodels.ImageModel(image_node)
    else:
        image_mod = image

    sca = int(image_mod.meta.instrument.detector[3:])
    date = astropy.time.Time(image_mod.meta.exposure.start_time)

    world_pos = astropy.coordinates.SkyCoord(
        image_mod.meta.wcsinfo.ra_ref * u.deg,
        image_mod.meta.wcsinfo.dec_ref * u.deg)

    if (distortion is None) and usecrds:
        import crds
        dist_name = crds.getreferences(
            image_mod.get_crds_parameters(),
            reftypes=['distortion'],
            observatory='roman',
        )['distortion']
        image_mod.meta.ref_file['distortion'] = os.path.basename(dist_name)

        dist_model = roman_datamodels.datamodels.DistortionRefModel(dist_name)
        distortion = dist_model.coordinate_distortion_transform

    if distortion is not None:
        wcs = make_wcs(util.skycoord(world_pos), distortion,
                       v2_ref=image_mod.meta.wcsinfo.v2_ref,
                       v3_ref=image_mod.meta.wcsinfo.v3_ref,
                       roll_ref=image_mod.meta.wcsinfo.roll_ref,
                       scale_factor=image_mod.meta.velocity_aberration.scale_factor)
        shape = image_mod.data.shape
        wcs.bounding_box = ((-0.5, shape[-1] - 0.5), (-0.5, shape[-2] - 0.5))
        wcs = GWCS(wcs)
    else:
        # use galsim.roman
        # galsim.roman does something smarter with choosing the roll
        # angle to optimize the solar panels given the target, and
        # therefore requires a date.
        wcs_dict = roman.getWCS(world_pos=util.celestialcoord(world_pos),
                                SCAs=sca,
                                date=date.datetime)
        wcs = wcs_dict[sca]
    return wcs


def make_wcs(targ_pos,
             distortion,
             roll_ref=0,
             v2_ref=0,
             v3_ref=0,
             wrap_v2_at=180,
             wrap_lon_at=360,
             scale_factor=1.0,
             ):
    """Create a gWCS from a target position, a roll, and a distortion map.

    Parameters
    ----------
    targ_pos : astropy.coordinates.SkyCoord
        The celestial coordinates of the boresight or science aperture.

    distortion : callable
        The distortion mapping pixel coordinates to V2/V3 coordinates for a
        detector.

    roll_ref : float
        The angle of the V3 axis relative to north, increasing from north to
        east, at the boresight or science aperture.
        Note that the V3 axis is rotated by +60 degree to the +Y axis.

    v2_ref : float
        The v2 coordinate (arcsec) corresponding to targ_pos

    v3_ref : float
        The v3 coordinate (arcsec) corresponding to targ_pos

    scale_factor : float
        The scale factor induced by velocity aberration

    Returns
    -------
    gwcs.wcs object representing WCS for observation
    """

    # it seems to me like the distortion mappings have v2_ref = v3_ref = 0,
    # which is easiest, so let me just keep those for now?
    # eventually to have greater ~realism, we'd want to set v2_ref and v3_ref
    # to whatever they'll end up being, different for each SCA.
    # We'd still need to get the ra_ref and dec_ref for each SCA using
    # this routine, though, with v2_ref = v3_ref = 0.  I need to think
    # a bit harder about whether we will also need to compute a separate
    # roll_ref for each SCA, and how that would best be done; if nothing else,
    # we do some finite differences to get the direction +V3 on the sky and
    # compute an angle wrt north.
    ra_ref = targ_pos.ra.to(u.deg).value
    dec_ref = targ_pos.dec.to(u.deg).value

    # v2_ref, v3_ref are in arcsec, but RotationSequence3D wants degrees,
    # so start by scaling by 3600.
    rot = models.RotationSequence3D(
        [v2_ref / 3600, -v3_ref / 3600, roll_ref, dec_ref, -ra_ref], 'zyxyz')

    # V2V3 are in arcseconds, while SphericalToCartesian expects degrees,
    # so again start by scaling by 3600
    tel2sky = ((models.Scale(1 / 3600) & models.Scale(1 / 3600))
               | gwcs.geometry.SphericalToCartesian(wrap_lon_at=wrap_v2_at)
               | rot
               | gwcs.geometry.CartesianToSpherical(wrap_lon_at=wrap_lon_at))
    tel2sky.name = 'v23tosky'

    detector = cf.Frame2D(name='detector', axes_order=(0, 1),
                          unit=(u.pix, u.pix))
    v2v3 = cf.Frame2D(name="v2v3", axes_order=(0, 1),
                      axes_names=("v2", "v3"), unit=(u.arcsec, u.arcsec))
    v2v3vacorr = cf.Frame2D(name='v2v3vacorr', axes_order=(0, 1),
                            axes_names=('v2', 'v3'), unit=(u.arcsec, u.arcsec))
    world = cf.CelestialFrame(reference_frame=astropy.coordinates.ICRS(),
                              name='world')

    # Compute differential velocity aberration (DVA) correction:
    va_corr = dva_corr_model(
        va_scale=scale_factor,
        v2_ref=v2_ref,
        v3_ref=v3_ref
    )

    pipeline = [gwcs.wcs.Step(detector, distortion),
                gwcs.wcs.Step(v2v3, va_corr),
                gwcs.wcs.Step(v2v3vacorr, tel2sky),
                gwcs.wcs.Step(world, None)]
    return gwcs.wcs.WCS(pipeline)


class GWCS(galsim.wcs.CelestialWCS):
    """This WCS uses gWCS to implent a galsim CelestialWCS.

    Based on galsim.fitswcs.AstropyWCS, edited to eliminate header functionality
    and to adopt the shared API supported by both gWCS and astropy.wcs.

    Parameters
    ----------
    gwcs : gwcs.WCS
        The WCS object to wrap in a galsim CelestialWCS interface.
    """

    def __init__(self, gwcs, origin=None):
        self._set_origin(origin)
        self._wcs = gwcs
        self._color = None

    @property
    def wcs(self):
        """The underlying ``gwcs.WCS`` object.
        """
        return self._wcs

    @property
    def origin(self):
        """The origin in image coordinates of the WCS function.
        """
        return self._origin

    def _radec(self, x, y, color=None):
        x1 = np.atleast_1d(x)
        y1 = np.atleast_1d(y)

        coord = self.wcs(x1, y1, with_bounding_box=False)
        r, d = (np.radians(c) for c in coord)

        if np.ndim(x) == np.ndim(y) == 0:
            return r[0], d[0]
        else:
            if (np.ndim(x) != np.ndim(y)):
                raise ValueError(
                    f"np.ndim(x) != np.ndim(y) => {np.ndim(x)} != {np.ndim(y)}")
            elif (x.shape != y.shape):
                raise ValueError(
                    f"x.shape != y.shape => {x.shape} != {y.shape}")
            return r, d

    def _xy(self, ra, dec, color=None):
        # _xy accepts ra/dec in radians; we decorate r1, d1 appropriately.
        r1 = np.atleast_1d(ra) * u.rad
        d1 = np.atleast_1d(dec) * u.rad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # x, y = self.wcs.world_to_pixel(r1, d1)
            x, y = self.wcs.numerical_inverse(r1, d1, with_bounding_box=False)

        if np.ndim(ra) == np.ndim(dec) == 0:
            return x[0], y[0]
        else:
            if (np.ndim(ra) != np.ndim(dec)):
                raise ValueError(
                    f"np.ndim(ra) != np.ndim(dec) => "
                    f"{np.ndim(ra)} != {np.ndim(dec)}")
            elif (x.shape != y.shape):
                raise ValueError(f"ra.shape != dec.shape => "
                                 f"{ra.shape} != {dec.shape}")
            return x, y

    def _newOrigin(self, origin):
        ret = self.copy()
        ret._origin = origin
        return ret

    def copy(self):
        ret = GWCS.__new__(GWCS)
        ret.__dict__.update(self.__dict__)
        return ret

    def __repr__(self):
        # tag = 'wcs=%r'%self.wcs
        tag = 'wcs=gWCS'  # gWCS repr strings can be very long.
        return "romanisim.wcs.GWCS(%s, origin=%r)" % (tag, self.origin)


def wcs_from_fits_header(header):
    """Convert a FITS WCS to a GWCS.

    This function reads SIP coefficients from a FITS WCS and implements
    the corresponding gWCS WCS.
    It was copied from gwcs.tests.utils._gwcs_from_hst_fits_wcs.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        FITS header

    Returns
    -------
    wcs : gwcs.wcs.WCS
        gwcs WCS corresponding to header
    """

    from astropy.modeling.models import (
        Shift, Polynomial2D, Pix2Sky_TAN, RotateNative2Celestial, Mapping)
    from astropy import wcs as fits_wcs

    # NOTE: this function ignores table distortions
    def coeffs_to_poly(mat, degree):
        pol = Polynomial2D(degree=degree)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if 0 < i + j <= degree:
                    setattr(pol, f'c{i}_{j}', mat[i, j])
        return pol

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fits_wcs.FITSFixedWarning)
        w = fits_wcs.WCS(header)
    ny, nx = 4089, 4089
    x0, y0 = w.wcs.crpix
    # original code has w.wcs.crpix - 1 here, but to match the _radec(...)
    # convention in the galsim CelestialWCS object, we delete that here.
    # Success is defined by
    # (GWCS._radec(x, y) ==
    #  wcs_from_fits_header(GWCS.header.header).pixel_to_world(x, y))

    cd = w.wcs.piximg_matrix

    cfx, cfy = np.dot(cd, [w.sip.a.ravel(), w.sip.b.ravel()])
    a = np.reshape(cfx, w.sip.a.shape)
    b = np.reshape(cfy, w.sip.b.shape)
    a[1, 0] = cd[0, 0]
    a[0, 1] = cd[0, 1]
    b[1, 0] = cd[1, 0]
    b[0, 1] = cd[1, 1]

    polx = coeffs_to_poly(a, w.sip.a_order)
    poly = coeffs_to_poly(b, w.sip.b_order)

    # construct GWCS:
    det2sky = (
        (Shift(-x0) & Shift(-y0)) | Mapping((0, 1, 0, 1)) | (polx & poly)
        | Pix2Sky_TAN() | RotateNative2Celestial(*w.wcs.crval, 180)
    )

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(
        reference_frame=getattr(astropy.coordinates, w.wcs.radesys).__call__(),
        name=w.wcs.radesys,
        unit=(u.deg, u.deg)
    )
    pipeline = [(detector_frame, det2sky), (sky_frame, None)]
    gw = gwcs.WCS(pipeline)
    gw.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

    return gw


def convert_wcs_to_gwcs(wcs):
    """Convert a GalSim WCS object into a GWCS object.

    Parameters
    ----------
    wcs : gwcs.wcs.WCS or wcs.GWCS
        input WCS to convert

    Returns
    -------
    wcs.GWCS corresponding to wcs.
    """
    if isinstance(wcs, GWCS):
        return wcs.wcs
    else:
        # make a gwcs WCS from a galsim.roman WCS
        return wcs_from_fits_header(wcs.header.header)


def get_mosaic_wcs(mosaic, shape=None, xpos=None, ypos=None, coord=None):
    """Get a WCS object for a given SCA or set of CRDS parameters.
    - if xpos, ypos, and coords are provided, then a GWCS compatible object will be created (and meta updated with it)
    - if not, a functional CelestialWCS is created [useful for quick computation,
    but GWCS needed for validation of a final simulation]

    Parameters
    ----------
    mosaic : roman_datamodels.datamodels.MosaicModel or dict
        Mosaic model or dictionary containing WCS parameters.
    shape: list
        Length of dimensions of mosaic
    xpos, ypos : array_like (float)
        x, y positions of each source in objlist
    coord : array_like (float)
        ra, dec positions of each source in objlist

    Returns
    -------
    galsim.CelestialWCS for the mosaic
    """

    # If sent a dictionary, create a temporary model for data interface
    if (type(mosaic) is not roman_datamodels.datamodels.MosaicModel):
        if shape is None:
            mosaic_node = maker_utils.mk_level3_mosaic()
        else:
            mosaic_node = maker_utils.mk_level3_mosaic(shape=shape)
        for key in mosaic.keys():
            if isinstance(mosaic[key], dict) and key in mosaic_node['meta'].keys():
                mosaic_node['meta'][key].update(mosaic[key])
            else:
                mosaic_node['meta'][key] = mosaic[key]
    else:
        mosaic_node = mosaic

    world_pos = astropy.coordinates.SkyCoord(
        mosaic_node.meta.wcsinfo.ra_ref * u.deg,
        mosaic_node.meta.wcsinfo.dec_ref * u.deg)

    if shape is None:
        shape = (mosaic_node.data.shape[0],
                 mosaic_node.data.shape[1])
        shape = mosaic_node.data.shape

    xcen = math.ceil(shape[1] / 2.0)
    ycen = math.ceil(shape[0] / 2.0)
    wcs = create_tangent_plane_gwcs(
        (xcen, ycen), romanisim.parameters.pixel_scale, world_pos)
    return wcs


def create_tangent_plane_gwcs(center, scale, center_coord):
    """Create a tangent plane GWCS object.

    Parameters
    ----------
    center : (float, float)
        pixel coordinates of center
    scale : float
        pixel scale (arcsec)
    center_coord : SkyCoord or CelestialCoord
        coordinates of center pixel
    
    Returns
    -------
    GWCS object corresponding to tangent plane projection
    """

    center_coord = util.skycoord(center_coord)
    pixelshift = models.Shift(-center[0]) & models.Shift(-center[1])
    pixelscale = models.Scale(scale / 3600.) & models.Scale(scale / 3600.)
    tangent_projection = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(
        center_coord.ra.to(u.deg).value,
        center_coord.dec.to(u.deg).value, 180.)
    det2sky = pixelshift | pixelscale | tangent_projection | celestial_rotation
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(
        reference_frame=astropy.coordinates.ICRS(), name='icrs',
        unit=(u.deg, u.deg))
    wcsobj = gwcs.wcs.WCS([(detector_frame, det2sky),
                          (sky_frame, None)])
    return wcsobj


def create_s_region(wcs, shape=None):
    """Create s_region string from wcs.

    Parameters
    ----------
    wcs : gwcs.wcs.WCS instance
        wcs for which s_region is desired
    shape : tuple
        use this shape to determine the pixel boundaries instead bounding box

    Returns
    -------
    s_region : str
        the s_region string, POLYGON ICRS + coordinates of 4 corners
    """
    if not isinstance(wcs, gwcs.wcs.WCS):
        raise ValueError('wcs must be a gwcs WCS object.')
    if shape is not None:
        bbox = ((-0.5, shape[-1] - 0.5), (-0.5, shape[-2] - 0.5))
    else:
        bbox = [[x for x in r] for r in wcs.bounding_box]
    bbox = [[int(round(r[0] + 0.5)), int(round(r[1] - 0.5))] for r in bbox]
    xcorn, ycorn = ([bbox[0][0], bbox[0][1], bbox[0][1], bbox[0][0]],
                    [bbox[1][0], bbox[1][0], bbox[1][1], bbox[1][1]])
    racorn, deccorn = wcs(xcorn, ycorn)
    rd = np.array([[r, d] for r, d in zip(racorn, deccorn)])
    s_region = "POLYGON ICRS " + " ".join([str(x) for x in rd.ravel()])
    return s_region


def dva_corr_model(va_scale, v2_ref, v3_ref):
    """
    Create transformation that accounts for differential velocity aberration
    (scale).

    Parameters
    ----------
    va_scale : float, None
        Ratio of the apparent plate scale to the true plate scale. When
        ``va_scale`` is `None`, it is assumed to be identical to ``1`` and
        an ``astropy.modeling.models.Identity`` model will be returned.

    v2_ref : float, None
        Telescope ``v2`` coordinate of the reference point in ``arcsec``. When
        ``v2_ref`` is `None`, it is assumed to be identical to ``0``.

    v3_ref : float, None
        Telescope ``v3`` coordinate of the reference point in ``arcsec``. When
        ``v3_ref`` is `None`, it is assumed to be identical to ``0``.

    Returns
    -------
    va_corr : astropy.modeling.CompoundModel, astropy.modeling.models.Identity
        A 2D compound model that corrects DVA. If ``va_scale`` is `None` or 1
        then `astropy.modeling.models.Identity` will be returned.

    """
    if va_scale is None or va_scale == 1:
        return models.Identity(2)

    if va_scale <= 0:
        raise ValueError("'Velocity aberration scale must be a positive number.")

    va_corr = models.Scale(va_scale, name="dva_scale_v2") & models.Scale(
        va_scale, name="dva_scale_v3"
    )

    if v2_ref is None:
        v2_ref = 0

    if v3_ref is None:
        v3_ref = 0

    if v2_ref == 0 and v3_ref == 0:
        return va_corr

    # NOTE: it is assumed that v2, v3 angles and va scale are small enough
    # so that for expected scale factors the issue of angle wrapping
    # (180 degrees) can be neglected.
    v2_shift = (1 - va_scale) * v2_ref
    v3_shift = (1 - va_scale) * v3_ref

    va_corr |= models.Shift(v2_shift, name="dva_v2_shift") & models.Shift(
        v3_shift, name="dva_v3_shift"
    )
    va_corr.name = "DVA_Correction"
    return va_corr
