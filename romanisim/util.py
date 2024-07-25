"""Miscellaneous utility routines.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import galsim

from romanisim import parameters, wcs
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

    if 'exposure' not in metadata.keys():
        metadata['exposure'] = {}
    read_pattern = metadata['exposure'].get(
        'read_pattern',
        parameters.read_pattern[metadata['exposure']['ma_table_number']])
    metadata['exposure']['read_pattern'] = read_pattern
    openshuttertime = parameters.read_time * read_pattern[-1][-1]
    offsets = dict(start=0 * u.s, mid=openshuttertime * u.s / 2,
                   end=openshuttertime * u.s)
    starttime = metadata['exposure']['start_time']
    if not isinstance(starttime, Time):
        starttime = Time(starttime, format='isot')
    for prefix, offset in offsets.items():
        metadata['exposure'][f'{prefix}_time'] = Time((
            starttime + offset).isot)
        metadata['exposure'][f'{prefix}_time_mjd'] = (
            starttime + offset).mjd
        metadata['exposure'][f'{prefix}_time_tdb'] = (
            starttime + offset).tdb.mjd
    metadata['exposure']['ngroups'] = len(read_pattern)
    metadata['exposure']['sca_number'] = (
        int(metadata['instrument']['detector'][-2:]))
    metadata['exposure']['integration_time'] = openshuttertime
    metadata['exposure']['elapsed_exposure_time'] = openshuttertime
    # ???
    metadata['exposure']['groupgap'] = 0
    metadata['exposure']['frame_time'] = parameters.read_time
    metadata['exposure']['exposure_time'] = openshuttertime
    metadata['exposure']['effective_exposure_time'] = openshuttertime
    metadata['exposure']['duration'] = openshuttertime
    # integration_start?  integration_end?  nints = 1?  ...

    if 'target' not in metadata.keys():
        metadata['target'] = {}
    target = metadata['target']
    target['type'] = 'FIXED'
    if 'wcsinfo' in metadata.keys():
        target['ra'] = metadata['wcsinfo']['ra_ref']
        target['dec'] = metadata['wcsinfo']['dec_ref']
        target['proposer_ra'] = target['ra']
        target['proposer_dec'] = target['dec']
    target['ra_uncertainty'] = 0
    target['dec_uncertainty'] = 0
    target['proper_motion_ra'] = 0
    target['proper_motion_dec'] = 0
    target['proper_motion_epoch'] = 'J2000'
    target['source_type'] = 'EXTENDED'

    # there are a few metadata keywords that have problematic, too-long
    # defaults in RDM.
    # program.category
    # ephemeris.ephemeris_reference_frame
    # guidestar.gs_epoch
    # this truncates these to the maximum allowed characters.  Alternative
    # solutions would include doing things like:
    #   making the roman_datamodels defaults archivable
    #   making the roman_datamodels validation check lengths of strings
    if 'program' in metadata:
        metadata['program']['category'] = metadata['program']['category'][:6]
    if 'ephemeris' in metadata:
        metadata['ephemeris']['ephemeris_reference_frame'] = (
            metadata['ephemeris']['ephemeris_reference_frame'][:10])
    if 'guidestar' in metadata:
        metadata['guidestar']['gs_epoch'] = (
            metadata['guidestar']['gs_epoch'][:10])


def update_aperture_and_wcsinfo_metadata(metadata, gwcs):
    """Update aperture and wcsinfo keywords to use the aperture for this SCA.

    Updates metadata in place, setting v2v3ref to be equal to the v2 and v3 of
    the center of the detector, and radecref accordingly.  Also updates the
    aperture to refer to this SCA.

    No updates are  performed if gwcs is not a gWCS object or if aperture and
    wcsinfo are not present in metadata.

    Parameters
    ----------
    metadata : dict
        Metadata to update
    gwcs : WCS object
        image WCS
    """
    if ('aperture' not in metadata or 'wcsinfo' not in metadata
            or not isinstance(gwcs, wcs.GWCS)):
        return
    gwcs = gwcs.wcs
    metadata['aperture']['name'] = (
        metadata['instrument']['detector'][:3] + '_'
        + metadata['instrument']['detector'][3:] + '_FULL')
    distortion = gwcs.get_transform('detector', 'v2v3')
    center = (galsim.roman.n_pix / 2 - 0.5, galsim.roman.n_pix / 2 - 0.5)
    v2v3 = distortion(*center)
    radec = gwcs(*center)
    metadata['wcsinfo']['ra_ref'] = radec[0]
    metadata['wcsinfo']['dec_ref'] = radec[1]
    metadata['wcsinfo']['v2_ref'] = v2v3[0]
    metadata['wcsinfo']['v3_ref'] = v2v3[1]


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


def default_image_meta(time=None, ma_table=1, filter_name='F087',
                       detector='WFI01', coord=None):
    """Return some simple default metadata for input to image.simulate

    Parameters
    ----------
    time : astropy.time.Time
        Time to use, default to 2020-01-01
    ma_table : int
        MA table number to use
    filter_name : str
        filter name to use
    detector : str
        detector to use
    coord : astropy.coordinates.SkyCoord
        coordinates to use, default to (270, 66)

    Returns
    -------
    Metadata dictionary corresponding to input parameters.
    """

    if time is None:
        time = Time('2020-01-01T00:00:00')
    if coord is None:
        coord = SkyCoord(270 * u.deg, 66 * u.deg)

    meta = {
        'exposure': {
            'start_time': time,
            'ma_table_number': 1,
        },
        'instrument': {
            'optical_element': filter_name,
            'detector': 'WFI01'
        },
        'wcsinfo': {
            'ra_ref': coord.ra.to(u.deg).value,
            'dec_ref': coord.dec.to(u.deg).value,
            'v2_ref': 0,
            'v3_ref': 0,
            'roll_ref': 0,
        },
    }

    return meta
