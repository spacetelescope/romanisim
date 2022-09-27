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
import crds
import asdf
import gwcs.geometry
from gwcs import coordinate_frames as cf
import gwcs.wcs
import galsim.wcs
from galsim import roman
from . import util


def get_wcs(world_pos, roll_ref=0, date=None, parameters=None, sca=None,
            usecrds=True):
    """Get a WCS object for a given sca or set of CRDS parameters.

    Parameters
    ----------
    world_pos : astropy.coordinates.SkyCoord or galsim.CelestialCoord
        boresight of telescope
    roll_ref : float
        roll of telescope V3 axis from North to East at boresight
    date : astropy.time.Time
        date of observation; used at least is usecrds = None to determine
        roll_ref.
    parameters : dict
        CRDS parameters dictionary specifying appropriate reference distortion
        map to load.
    sca : int
        WFI sensor chip array number
    usecrds : bool
        If True, use crds reference distortions rather than galsim.roman
        distortion model.

    Returns
    -------
    galsim.CelestialWCS for an SCA
    """

    from .parameters import default_parameters_dictionary

    if parameters is not None:
        parameters = util.flatten_dictionary(parameters)
    if parameters is None and sca is None:
        raise ValueError('At least one of parameters or sca must be set!')
    if parameters is None:
        from copy import deepcopy
        parameters = deepcopy(default_parameters_dictionary)
        parameters = util.flatten_dictionary(parameters)
        parameters['roman.meta.instrument.detector'] = 'WFI%02d' % sca
    elif sca is None:
        sca = int(parameters['roman.meta.instrument.detector'][3:])
    if date is None:
        date = Time(parameters['roman.meta.exposure.start_time'],
                    format='isot')
    if usecrds:
        fn = crds.getreferences(parameters, reftypes=['distortion'],
                                observatory='roman')
        distortion = asdf.open(fn['distortion'])
        wcs = make_wcs(
            util.skycoord(world_pos), roll_ref,
            distortion['roman']['coordinate_distortion_transform'])
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


def make_wcs(targ_pos, roll_ref, distortion, wrap_v2_at=180, wrap_lon_at=360):
    """Create a gWCS from a target position, a roll, and a distortion map.

    Parameters
    ----------
    targ_pos : astropy.coordinates.SkyCoord
        The celestial coordinates of the boresight.

    roll_ref : float
        The angle of the V3 axis relative to north, increasing from north to
        east, at the boresight.

    distortion : callable
        The distortion mapping pixel coordinates to V2/V3 coordinates for a
        detector.

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
    v2_ref = v3_ref = 0
    ra_ref = targ_pos.ra.to(u.deg).value
    dec_ref = targ_pos.dec.to(u.deg).value

    # full transformation from romancal.assign_wcs.pointing
    # angles = np.array([v2_ref, -v3_ref, roll_ref, dec_ref, -ra_ref])
    # axes = "zyxyz"
    # rot = RotationSequence3D(angles, axes_order=axes)
    from astropy.modeling.models import RotationSequence3D, Scale
    rot = RotationSequence3D([roll_ref, dec_ref, -ra_ref], 'xyz')

    # distortion takes pixels to V2V3
    # V2V3 are in arcseconds, while SphericalToCartesian expects degrees.
    model = distortion | ((Scale(1 / 3600) & Scale(1 / 3600)) |
        gwcs.geometry.SphericalToCartesian(wrap_lon_at=wrap_v2_at)
         | rot | gwcs.geometry.CartesianToSpherical(wrap_lon_at=wrap_lon_at))
    model.name = 'pixeltosky'
    detector = cf.Frame2D(name='detector', axes_order=(0, 1),
                          unit=(u.pix, u.pix))
    world = cf.CelestialFrame(reference_frame=astropy.coordinates.ICRS(),
                              name='world')
    return gwcs.wcs.WCS(model, input_frame=detector, output_frame=world)


class GWCS(galsim.wcs.CelestialWCS):
    """This WCS uses gWCS to implent a galsim CelestialWCS.

    GWCS is initialized via

       >>> wcs = romanisim.wcs.GWCS(gwcs)

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
            assert np.ndim(x) == np.ndim(y)
            assert x.shape == y.shape
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
            assert np.ndim(ra) == np.ndim(dec)
            assert ra.shape == dec.shape
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
        return "romanisim.wcs.GWCS(%s, origin=%r)"%(tag, self.origin)
