"""Roman WCS interface for galsim.

galsim.roman has an implementation of Roman's WCS based on some SIP
coefficients for each SCA.  This is presumably plenty good, but here
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

import warnings
import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy.modeling import models
import astropy.time
import roman_datamodels
import crds
import gwcs.geometry
from gwcs import coordinate_frames as cf
import gwcs.wcs
import galsim.wcs
from galsim import roman
from . import util


# Needed until RCAL release unfreezing link to RDM/RAD versions 0.14.1
try:
    import roman_datamodels.maker_utils as maker_utils
except ImportError:
    import roman_datamodels.testing.utils as maker_utils



def fill_in_parameters(parameters, coord, roll_ref=0, boresight=True):
    """Add WCS info to parameters dictionary.

    Parameters
    ----------
    parameters : dict
        CRDS parameters dictionary
        keys like pointing.* and wcsinfo.* may be modified

    coord : astropy.coordinates.SkyCoord or galsim.CelestialCoord
        world coordinates at V2 / V3 ref (boresight or center of WFI CCDs)

    roll_ref : float
        roll of the V3 axis from north

    boresight : bool
        whether coord is the telescope boresight (V2 = V3 = 0) or the center of
        the WFI CCD array
    """
    coord = util.skycoord(coord)

    if 'pointing' not in parameters.keys():
        parameters['pointing'] = {}

    parameters['pointing']['ra_v1'] = coord.ra.to(u.deg).value

    if 'wcsinfo' not in parameters.keys():
        parameters['wcsinfo'] = {}

    parameters['wcsinfo']['ra_ref'] = (
        parameters['pointing']['ra_v1'])

    parameters['pointing']['dec_v1'] = coord.dec.to(u.deg).value
    parameters['wcsinfo']['dec_ref'] = (
        parameters['pointing']['dec_v1'])

    parameters['wcsinfo']['roll_ref'] = roll_ref

    if boresight:
        parameters['wcsinfo']['v2_ref'] = 0
        parameters['wcsinfo']['v3_ref'] = 0
    else:
        from .parameters import v2v3_wficen
        v2_ref = v2v3_wficen[0] / 3600
        v3_ref = v2v3_wficen[1] / 3600
        parameters['wcsinfo']['v2_ref'] = v2_ref
        parameters['wcsinfo']['v3_ref'] = v3_ref
        parameters['wcsinfo']['roll_ref'] = (
            parameters['wcsinfo'].get('roll_ref', 0) + 60)

def get_wcs(image, usecrds=True, distortion=None):
    """Get a WCS object for a given sca or set of CRDS parameters.

    Parameters
    ----------
    image : roman_datamodels.datamodels.ImageModel or dict
        Image model or dictionary containing CRDS parameters
        specifying appropriate reference distortion
        map to load.
    usecrds : bool
        If True, use crds reference distortions rather than galsim.roman
        distortion model.
    boresight : bool
        If True, world_pos specifies the location of the telescope boresight;
        otherwise the location of the science aperture.

    Returns
    -------
    galsim.CelestialWCS for an SCA
    """

    # If sent a dictionary, create a temporary model for CRDS interface
    if(type(image) != roman_datamodels.datamodels.ImageModel):
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
        dist_name = crds.getreferences(
            image_mod.get_crds_parameters(),
            reftypes=['distortion'],
            observatory='roman',
        )['distortion']

        dist_model = roman_datamodels.datamodels.DistortionRefModel(dist_name)
        distortion = dist_model.coordinate_distortion_transform

    if distortion is not None:
        wcs = make_wcs(util.skycoord(world_pos), distortion,
                       v2_ref=image_mod.meta.wcsinfo.v2_ref,
                       v3_ref=image_mod.meta.wcsinfo.v3_ref,
                       roll_ref=image_mod.meta.wcsinfo.roll_ref)
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


def make_wcs(targ_pos, distortion, roll_ref=0, v2_ref=0, v3_ref=0,
             wrap_v2_at=180, wrap_lon_at=360):
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

    # full transformation from romancal.assign_wcs.pointing
    # angles = np.array([v2_ref, -v3_ref, roll_ref, dec_ref, -ra_ref])
    # axes = "zyxyz"
    # rot = RotationSequence3D(angles, axes_order=axes)
    rot = models.RotationSequence3D(
        [v2_ref, -v3_ref, roll_ref, dec_ref, -ra_ref], 'zyxyz')

    # distortion takes pixels to V2V3
    # V2V3 are in arcseconds, while SphericalToCartesian expects degrees.
    model = (distortion | (models.Scale(1 / 3600) & models.Scale(1 / 3600))
             | gwcs.geometry.SphericalToCartesian(wrap_lon_at=wrap_v2_at)
             | rot
             | gwcs.geometry.CartesianToSpherical(wrap_lon_at=wrap_lon_at))
    model.name = 'pixeltosky'
    detector = cf.Frame2D(name='detector', axes_order=(0, 1),
                          unit=(u.pix, u.pix))
    world = cf.CelestialFrame(reference_frame=astropy.coordinates.ICRS(),
                              name='world')
    return gwcs.wcs.WCS(model, input_frame=detector, output_frame=world)


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

        coord = self.wcs.pixel_to_world(x1, y1)
        r, d = coord.ra.to(u.rad).value, coord.dec.to(u.rad).value

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
            x, y = self.wcs.world_to_pixel(r1, d1)

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
