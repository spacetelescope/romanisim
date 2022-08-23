import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import galsim


def skycoord(celestial):
    """Turn a CelestialCoord into a SkyCoord.

    Params
    ------
    celestial : galsim.CelestialCoord
        galsim.CelestialCoord to transform into an astropy.coordinates.SkyCoord
    
    Returns
    -------
    astropy.coordinates.SkyCoord
        SkyCoord corresponding to celestial
    """
    if isinstance(celestial, SkyCoord):
        return celestial
    else:
        return SkyCoord(ra=(celestial.ra / galsim.radians)*u.rad,
                        dec=(celestial.dec / galsim.radians)*u.rad,
                        frame='icrs')


def celestialcoord(sky):
    """Turn a SkyCoord into a CelestialCoord.

    Params
    ------
    sky : astropy.coordinates.SkyCoord
        astropy.coordinates.SkyCoord to transform into an galsim.CelestialCoord
    
    Returns
    -------
    galsim.CelestialCoord
        CelestialCoord corresponding to skycoord
    """
    if isinstance(sky, galsim.CelestialCoord):
        return sky
    else:
        return galsim.CelestialCoord(sky.ra.to(u.rad).value*galsim.radians,
                                     sky.dec.to(u.rad).value*galsim.radians)


def scalergb(rgb, scales=None, lumrange=None):
    rgb = rgb.copy()
    if scales is not None:
        for i in range(3):
            rgb[:, :, i] /= scales[i]
    norm = np.sqrt(rgb[:, :, 0]**2 + rgb[:, :, 1]**2 + rgb[:, :, 2]**2)
    if lumrange is None:
        lumrange = [0, np.max(norm)]
    newnorm = np.clip((norm - lumrange[0]) / (lumrange[1] - lumrange[0]),
                      0, 1)
    out = rgb.copy()
    for i in range(3):
        out[:, :, i] = out[:, :, i] * newnorm / norm
    return out


def random_points_in_cap(coord, radius, nobj, rng=None):
    """Choose locations at random within radius of coord.

    Params
    ------
    coord : astropy.coordinates.SkyCoord
        location around which to generate points

    radius : float
        radius in deg of region in which to generate points

    nobj : int
        number of objects to generate

    rng : galsim.UniformDeviate
        random number generator to use

    Returns
    -------
    astropy.coordinates.SkyCoord

    """
    if rng is None:
        rng = galsim.UniformDeviate()

    ang = np.zeros(nobj, dtype='f8')
    dist = np.zeros(nobj, dtype='f8')
    rng.generate(ang)
    rng.generate(dist)
    ang *= 2*np.pi
    dist = np.arccos(1-(1-np.cos(np.radians(radius)))*dist)
    c1 = SkyCoord(coord.ra.rad*u.rad, coord.dec.rad*u.rad, frame='icrs')
    c1 = c1.directional_offset_by(ang*u.rad, dist*u.rad)
    sky_pos = [celestialcoord(x) for x in c1]
    return sky_pos
