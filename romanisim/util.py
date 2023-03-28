"""Miscellaneous utility routines.
"""

import datetime
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import galsim
from roman_datamodels import stnode
from romanisim import parameters
from scipy import integrate


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
        return SkyCoord(ra=(celestial.ra / galsim.radians) * u.rad,
                        dec=(celestial.dec / galsim.radians) * u.rad,
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
        return galsim.CelestialCoord(sky.ra.to(u.rad).value * galsim.radians,
                                     sky.dec.to(u.rad).value * galsim.radians)


def scalergb(rgb, scales=None, lumrange=None):
    """Scales three flux images into a range of luminosity for displaying.

    Images are scaled into [0, 1].

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

    rgb = np.clip(rgb, 0, np.inf)
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
        out[:, :, i] = out[:, :, i] * newnorm / (norm + (norm == 0))
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

    dist = np.zeros(nobj, dtype='f8')
    rng.generate(dist)
    dist = np.arccos(1 - (1 - np.cos(np.radians(radius))) * dist) * u.rad
    return random_points_at_radii(coord, dist, rng=rng)


def random_points_in_king(coord, rc, rt, nobj, rng=None):
    """Sample points from a King distribution

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        location around which to generate points
    rc : float
        core radius in deg
    rt : float
        truncation radius in deg
    nobj : int
        number of objects to generate
    rng : galsim.UniformDeviate
        random number generator to use

    Returns
    -------
    astropy.coordinates.SkyCoord
    """
    distances = sample_king_distances(rt, rt, nobj, rng=rng) * u.deg
    return random_points_at_radii(coord, distances, rng=rng)


def random_points_at_radii(coord, radii, rng=None):
    """Choose locations at random at given radii from coord.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        location around which to generate points
    distances : astropy.Quantity[float]
        angular distances points should lie from center
    rng : galsim.UniformDeviate
        random number generator to use

    Returns
    -------
    astropy.coordinates.SkyCoord

    """
    if rng is None:
        rng = galsim.UniformDeviate()

    ang = np.zeros(len(radii), dtype='f8')
    rng.generate(ang)
    ang *= 2 * np.pi
    c1 = SkyCoord(coord.ra.rad * u.rad, coord.dec.rad * u.rad, frame='icrs')
    c1 = c1.directional_offset_by(ang * u.rad, radii)
    return c1


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
                out[key + '.' + subkey] = subvalue
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
        except Exception:
            return v
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


def add_more_metadata(metadata):
    """Fill out the metadata dictionary, modifying it in place.

    Parameters
    ----------
    metadata : dict
        CRDS-style dictionary containing keywords like
        roman.meta.exposure.start_time.
    """

    # fill out the metadata a bit with redundant stuff for which we
    # already mostly have the answer.
    ma_table = parameters.ma_table[
        metadata['roman.meta.exposure.ma_table_number']]
    openshuttertime = parameters.read_time * (
        ma_table[-1][0] + ma_table[-1][1] - 1)
    offsets = dict(start=0 * u.s, mid=openshuttertime * u.s / 2,
                   end=openshuttertime * u.s)
    starttime = metadata['roman.meta.exposure.start_time']
    if not isinstance(starttime, Time):
        starttime = Time(starttime, format='isot')
    for prefix, offset in offsets.items():
        metadata[f'roman.meta.exposure.{prefix}_time'] = (
            starttime + offset).isot
        metadata[f'roman.meta.exposure.{prefix}_time_mjd'] = (
            starttime + offset).mjd
        metadata[f'roman.meta.exposure.{prefix}_time_tdb'] = (
            starttime + offset).tdb.mjd
    metadata['roman.meta.exposure.ngroups'] = len(ma_table)
    metadata['roman.meta.exposure.nframes'] = ma_table[0][0]
    metadata['roman.meta.exposure.sca_number'] = (
        int(metadata['roman.meta.instrument.detector'][-2:]))
    metadata['roman.meta.exposure.integration_time'] = openshuttertime
    metadata['roman.meta.exposure.elapsed_exposure_time'] = openshuttertime
    # ???
    metadata['roman.meta.exposure.frame_divisor'] = ma_table[0][1]
    metadata['roman.meta.exposure.groupgap'] = 0
    metadata['roman.meta.exposure.frame_time'] = parameters.read_time
    metadata['roman.meta.exposure.group_time'] = (
        parameters.read_time * ma_table[0][1])
    metadata['roman.meta.exposure.exposure_time'] = openshuttertime
    metadata['roman.meta.exposure.effective_exposure_time'] = openshuttertime
    metadata['roman.meta.exposure.duration'] = openshuttertime
    # integration_start?  integration_end?  nints = 1?  ...


def king_profile(r, rc, rt):
    """Compute the King (1962) profile.

    Parameters
    ----------
    r : np.ndarray[float]
        distances at which to evaluate the King profile
    rc : float
        core radius
    rt : float
        truncation radius

    Returns
    -------
        2D number density of stars at r.
    """
    return (1 / np.sqrt(1 + (r / rc)**2) - 1 / np.sqrt(1 + (rt / rc)**2))**2


def sample_king_distances(rc, rt, npts, rng=None):
    """Sample distances from a King (1962) profile.

    Parameters
    ----------
    rc : float
        core radius
    rt : float
        truncation radius
    npts : int
        number of points to generate
    rng : galsim.BaseDeviate
        random number generator to use

    Returns
    -------
    r : float
        Distances distributed according to a King (1962) profile.
    """
    rng = galsim.UniformDeviate(rng)
    rr = np.zeros(npts, dtype='f4')
    rng.generate(rr)
    logx = np.linspace(np.log(rc) - 4, np.log(rt), 1000)
    x = np.concatenate([[0], np.exp(logx)])
    pdf = king_profile(x, rc, rt)
    cdf = integrate.cumulative_trapezoid(pdf * x, x, initial=0)
    cdf /= cdf[-1]
    radii = np.interp(rr, cdf, x)
    return radii
