"""Miscellaneous utility routines.
"""

import datetime
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import galsim
from roman_datamodels import stnode


def skycoord(celestial):
    """Turn a CelestialCoord into a SkyCoord.

    Parameters
    ----------
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

    Parameters
    ----------
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
    """Scales three flux images into a range of luminosity for displaying.

    This routine is intended to help with cases where you want to display
    some images and want the color scale to cover only a certain range,
    but saturated regions should retain their appropriate hue and not be
    compressed to white.

    Parameters
    ----------
    rgb : np.ndarray[npix, npix, 3]
        the RGB images to scale
    scales : list[float] (must contain 3 floats)
        rescale each image by this amount
    lumrange : list[float] (must contain 2 floats)
        minimum and maximum luminosity

    Returns
    -------
    im : np.ndarray[npix, npix, 3]
        scaled RGB image suitable for displaying
    """

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

    Parameters
    ----------
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


def flatten_dictionary(d):
    """Convert a set of nested dictionaries into a flattened dictionary.

    Some routines want dictionaries of the form dict[key1][key2][key3], while
    others want dict[key1.key2.key3].  This function converts the former into
    the latter.

    This can do the wrong thing in cases that don't make sense, e.g., if
    the top level dictionary contains keys including dots that overlap with
    the names of keys that this function would like to make.  e.g.:
    {'a.b': 1, 'a': {'b': 2}}.

    This code is garbage that should be replaced with some better handling of
    the CRDS <-> ASDF metadata transformations, but I don't fully understand
    what's happening there and this can stand in as a placeholder.

    Parameters
    ----------
    d : dict
        dictionary to flatten

    Returns
    -------
    dict
        flattened dictionary, with subdictionaries' keys promoted into the
        top-level directory with keys adjusted to include dots indicating
        their former position in the hierarchy.
    """
    out = dict()
    for key, value in d.items():
        if isinstance(value, dict):
            flattened = flatten_dictionary(value)
            for subkey, subvalue in flattened.items():
                out[key+'.'+subkey] = subvalue
        else:
            flatval = value
            if isinstance(flatval, Time):
                flatval = str(value)
            elif isinstance(flatval, datetime.datetime):
                flatval = value.isoformat()
            out[key] = flatval
    return out


def unflatten_dictionary(d):
    """Convert a flattened dictionary into a set of nested dictionaries.

    Some routines want dictionaries of the form dict[key1][key2][key3], while
    others want dict[key1.key2.key3].  This functions converts the latter into
    the former.

    This code is garbage that should be replaced with some better handling of
    the CRDS <-> ASDF metadata transformations, but I don't fully understand
    what's happening there and this can stand in as a placeholder.

    Parameters
    ----------
    d : dict
        dictionary to unflatten

    Returns
    -------
    dict
        unflattened dictionary, with keys with dots promoted into subdictionaries.
    """

    def unflatten_value(k, v):
        try:
            v = Time(v, format='isot')
            if 'file' in k:
                v = stnode.FileDate(v)
        except Exception as e:
            pass
        return v

    out = dict()
    for key, value in d.items():
        subdicts = key.split('.')
        if len(subdicts) == 1:
            out[key] = unflatten_value(key, value)
            continue
        tdict = out
        for subdict0 in subdicts[:-1]:
            tsubdict = tdict.get(subdict0, None)
            if tsubdict is None:
                tsubdict = dict()
                tdict[subdict0] = tsubdict
            tdict = tsubdict
        tdict[subdicts[-1]] = unflatten_value(key, value)
    return out
