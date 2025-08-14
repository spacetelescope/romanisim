"""Miscellaneous utility routines.
"""

import numpy as np
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy import units as u
from astropy.time import Time
import galsim
import gwcs as gwcsmod

from romanisim import parameters, wcs, bandpass
from romanisim.velocity_aberration import compute_va_effects
from scipy import integrate
from collections.abc import Mapping

__all__ = ["skycoord",
           "celestialcoord",
           "scalergb",
           "random_points_in_cap",
           "random_points_in_king",
           "random_points_at_radii",
           "add_more_metadata",
           "update_pointing_and_wcsinfo_metadata",
           "king_profile",
           "sample_king_distances",
           "decode_context_times",
           "default_image_meta",
           "update_photom_keywords",
           "merge_dicts",
           "calc_scale_factor",
]


def skycoord(celestial):
    """Turn a CelestialCoord into a SkyCoord.

    Parameters
    ----------
    celestial : galsim.CelestialCoord
        galsim.CelestialCoord to transform into an astropy.coordinates.SkyCoord

    Returns
    -------
    coord: astropy.coordinates.SkyCoord
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
        astropy.coordinates.SkyCoord to transform into a galsim.CelestialCoord

    Returns
    -------
    coord: galsim.CelestialCoord
        CelestialCoord corresponding to skycoord
    """
    if isinstance(sky, galsim.CelestialCoord):
        return sky
    else:
        return galsim.CelestialCoord(sky.ra.to(u.rad).value * galsim.radians,
                                     sky.dec.to(u.rad).value * galsim.radians)


def scalergb(rgb, scales=None, gray=0):
    """Scales three flux images into an RGB image for displaying.

    Images are scaled into [0, 1].  This routine was borrowed from the Legacy
    Survey viewer.  It expects background subtracted images.

    Parameters
    ----------
    rgb : np.ndarray[ny, nx, 3]
        the RGB images to scale
    scales : list[float] (must contain 3 floats)
        rescale each image by this amount
    gray : float
        add a constant to the image to shift the sky background value from 0 to gray

    Returns
    -------
    im : np.ndarray[ny, nx, 3]
        scaled RGB image suitable for displaying
    """

    rgb = rgb.copy()
    if scales is not None:
        for i in range(3):
            rgb[:, :, i] = rgb[:, :, i] / scales[i] + gray
    intensity = np.sum(np.clip(rgb, 0, np.inf), axis=2)  # how much light is there?
    Q = 20  # taken from legacyviewer
    fI = np.arcsinh(Q * intensity) / np.sqrt(Q)
    intensity += (intensity == 0) * 1e-6
    rgb *= (fI / intensity)[:, :, None]
    rgb = np.clip(rgb, 0, 1)
    return rgb


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
    coords : astropy.coordinates.SkyCoord
        Coordinates selected at random in cap.
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
    coords : astropy.coordinates.SkyCoord
        sample coordinates selected
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
    coords : astropy.coordinates.SkyCoord
        random coordinates selected
    """
    if rng is None:
        rng = galsim.UniformDeviate()

    ang = np.zeros(len(radii), dtype='f8')
    rng.generate(ang)
    ang *= 2 * np.pi
    c1 = SkyCoord(coord.ra.rad * u.rad, coord.dec.rad * u.rad, frame='icrs')
    c1 = c1.directional_offset_by(ang * u.rad, radii)
    return c1


def add_more_metadata(metadata, usecrds=False):
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
    if 'guide_star' not in metadata.keys():
        metadata['guide_star'] = {}
    manum = metadata['exposure']['ma_table_number']
    metadata['exposure']['ma_table_id'] = f'SCI{manum:04d}'

    if getattr(parameters, 'ma_table_reference', None):
        matab = parameters.ma_table_reference
    
    if usecrds:
        metadata['exposure']['frame_time'] = matab['roman']['science_tables'][f'SCI{manum:04}']['frame_time']

        read_pattern = metadata['exposure'].get(
            'read_pattern',
            matab['roman']['science_tables'][f'SCI{manum:04}']['science_read_pattern'][nresultants-1])
        openshuttertime = metadata['exposure']['frame_time'] * read_pattern[-1][-1]

        metadata['exposure']['exposure_time'] = matab['roman']['science_tables'][f'SCI{manum:04}']['accumulated_exposure_time'][nresultants-1]
        metadata['exposure']['effective_exposure_time'] = matab['roman']['science_tables'][f'SCI{manum:04}']['effective_exposure_time'][nresultants-1]
    else:
        metadata['exposure']['frame_time'] = parameters.read_time

        read_pattern = metadata['exposure'].get(
            'read_pattern',
            parameters.read_pattern[metadata['exposure']['ma_table_number']])
        openshuttertime = parameters.read_time * read_pattern[-1][-1]

        metadata['exposure']['exposure_time'] = round(openshuttertime, 4)
        effexptime = parameters.read_time * (np.mean(read_pattern[-1]))        
        metadata['exposure']['effective_exposure_time'] = round(effexptime, 4)
    metadata['exposure']['nresultants'] = len(read_pattern)

    offsets = dict(start=0 * u.s, mid=openshuttertime * u.s / 2,
                end=openshuttertime * u.s)
    starttime = metadata['exposure']['start_time']
    if not isinstance(starttime, Time):
        starttime = Time(starttime, format='isot')
    for prefix, offset in offsets.items():
        metadata['exposure'][f'{prefix}_time'] = Time((
            starttime + offset).isot)

    if 'window_xstart' in metadata['guide_star']:
        metadata['guide_star']['window_xstop'] = (
            metadata['guide_star']['window_xstart'] + 16)
        metadata['guide_star']['window_ystop'] = (
            metadata['guide_star']['window_ystart'] + 16)
    if 'visit' not in metadata.keys():
        metadata['visit'] = dict()
    metadata['visit']['status'] = 'SUCCESSFUL'



def update_pointing_and_wcsinfo_metadata(metadata, gwcs):
    """Update pointing and wcsinfo keywords to use the aperture for this SCA.

    Updates metadata in place, setting v2/v3_ref to be equal to the V2 and V3 of
    the center of the detector, and ra/dec_ref accordingly.  Also updates the
    pointing to refer to this SCA and ra/dec_v1 to point along the boresight.

    No updates are  performed if gwcs is not a gWCS object or if pointing and
    wcsinfo are not present in metadata.

    Parameters
    ----------
    metadata : dict
        Metadata to update
    gwcs : WCS object
        image WCS
    """
    if 'pointing' not in metadata or 'wcsinfo' not in metadata:
        return
    if isinstance(gwcs, wcs.GWCS):
        gwcs = gwcs.wcs
    if not isinstance(gwcs, gwcsmod.wcs.WCS):
        return
    metadata['wcsinfo']['aperture_name'] = (
        metadata['instrument']['detector'] + '_FULL')
    distortion = gwcs.get_transform('detector', 'v2v3')
    center = (galsim.roman.n_pix / 2 - 0.5, galsim.roman.n_pix / 2 - 0.5)
    v2v3 = distortion(*center)
    radec = gwcs(*center)
    t2sky = gwcs.get_transform('v2v3', 'world')
    radecn = t2sky(v2v3[0], v2v3[1] + 100)
    roll_ref = (
        SkyCoord(radec[0] * u.deg, radec[1] * u.deg).position_angle(
        SkyCoord(radecn[0] * u.deg, radecn[1] * u.deg)))
    roll_ref = roll_ref.to(u.deg).value
    # new roll ref appropriate for new v2v3 ref
    # note: some of this logic will need to change after
    # aberration is incorporated.  roll_ref will need to
    # be computed from the aberrated v2v3 frame to the sky.
    # but maybe this doesn't matter since the aberrated v2v3
    # axes are parallel to the v2v3 axes.
    # whether we need to change radec ref, v2v3 ref depends
    # on how we ultimately define these quantities.

    metadata['wcsinfo']['ra_ref'] = radec[0]
    metadata['wcsinfo']['dec_ref'] = radec[1]
    metadata['wcsinfo']['v2_ref'] = v2v3[0]
    metadata['wcsinfo']['v3_ref'] = v2v3[1]
    metadata['wcsinfo']['roll_ref'] = roll_ref

    boresight = t2sky(0, 0)
    metadata['pointing']['ra_v1'] = boresight[0]
    metadata['pointing']['dec_v1'] = boresight[1]
    boresightn = t2sky(0, 1)
    pa_v3 = (
        SkyCoord(boresight[0] * u.deg, boresight[1] * u.deg).position_angle(
        SkyCoord(boresightn[0] * u.deg, boresightn[1] * u.deg)))
    pa_v3 = pa_v3.to(u.deg).value
    metadata['pointing']['pa_v3'] = pa_v3

    # Update velocity aberration meta for the reference point
    metadata['velocity_aberration']['ra_reference'] = radec[0]
    metadata['velocity_aberration']['dec_reference'] = radec[1]


def king_profile(r, rc, rt):
    """Compute the King profile.

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
    rho : np.ndarray[float]
        2D number density of stars at r.
    """
    return (1 / np.sqrt(1 + (r / rc)**2) - 1 / np.sqrt(1 + (rt / rc)**2))**2


def sample_king_distances(rc, rt, npts, rng=None):
    """Sample distances from a King profile.

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


def decode_context_times(context, exptimes):
    """
    Get 0-based indices of input images that contributed to (resampled)
    output pixel with coordinates ``x`` and ``y``.

    Parameters
    ----------
    context: numpy.ndarray
        A 3D numpy.ndarray of integral data type.

    exptimes: list of floats
        Exposure times for each component image.


    Returns
    -------
    total_exptimes : numpy.ndarray
        A 2D array of total exposure time for each pixel.
    """

    if context.ndim != 3:
        raise ValueError("'context' must be a 3D array.")

    """
    Context decoding example:
    An example context array for an output image of array shape ``(5, 6)``
    obtained by resampling 80 input images.

    .. code-block:: none

        con = np.array(
            [[[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 36196864, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 537920000, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0,],
              [0, 0, 0, 67125536, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 163856, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 8203, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 32865, 0, 0, 0]]],
            dtype=np.int32
        )
        decode_context(con, [3, 2], [1, 4])
        [array([ 9, 12, 14, 19, 21, 25, 37, 40, 46, 58, 64, 65, 67, 77]),
        array([ 9, 20, 29, 36, 47, 49, 64, 69, 70, 79])]
    """

    nbits = 8 * context.dtype.itemsize

    total_exptimes = np.zeros(context.shape[1:])

    for y in range(total_exptimes.shape[0]):
        for x in range(total_exptimes.shape[1]):
            files = [v & (1 << k) for v in context[:, y, x] for k in range(nbits)]
            tot_time = 0
            files = [file for file in files if (file != 0)]

            for im_idx in files:
                tot_time += exptimes[im_idx - 1]

            total_exptimes[y,x] = tot_time

    def sum_times(x):
        tot_time = 0
        files = [x & (1 << k) for k in range(nbits)]
        files = [file for file in files if (file != 0)]
        for im_idx in files:
            tot_time += exptimes[im_idx - 1]
        return tot_time

    vectorized_sum_times = np.vectorize(sum_times)

    total_exptimes = vectorized_sum_times(context[:,])
    total_exptimes = total_exptimes.reshape(total_exptimes.shape[1:])

    return total_exptimes


def default_image_meta(time=None, ma_table=4, filter_name='F087',
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
    meta : dict
        Metadata dictionary corresponding to input parameters.
    """

    if time is None:
        time = Time('2020-01-01T00:00:00')
    if coord is None:
        coord = SkyCoord(270 * u.deg, 66 * u.deg)

    meta = {
        'exposure': {
            'start_time': time,
            'ma_table_number': 4,
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


def update_photom_keywords(im, gain=None):
    """Add photometry calibration keywords to image metadata.

    This fills out the im.meta['photometry'] keywords:

    * conversion_megajanskys (MJy/sr corresponding to 1 DN / s / pix)
    * conversion_microjanskys (uJy/sq. arcsec corresponding to 1 DN / s / pix)
    * corresponding uncertainties (0 in appropriate units)
    * pixelarea_steradians (area of central pixel in steradians)
    * pixelarea_arcsecsq (area of central pixel in square arcseconds)

    The values are derived from the bandpasses of the filters and and the WCS
    of the image.

    Parameters
    ----------
    im : roman_datamodels.ImageModel
        Image whose metadata should be updated with photometry keywords
    gain : float, Quantity, array
        Gain image to use
    """
    gain = (np.median(gain)
            if gain is not None else parameters.reference_data['gain'])
    gain = gain.value if isinstance(gain, u.Quantity) else gain
    if 'wcs' in im['meta']:
        wcs = im['meta']['wcs']
        cenpix = (im.data.shape[0] // 2, im.data.shape[1] // 2)
        cc = wcs.pixel_to_world(
            (cenpix[0], cenpix[0], cenpix[0] + 1),
            (cenpix[1], cenpix[1] + 1, cenpix[1]))
        angle = (cc[0].position_angle(cc[1]) -
                 cc[0].position_angle(cc[2]))
        area = (cc[0].separation(cc[1]) * cc[0].separation(cc[2])
                * np.sin(angle.to(u.rad).value))
        im['meta']['photometry']['pixel_area'] = area.to(u.sr).value
        val = (gain * (3631 / bandpass.get_abflux(
             im.meta['instrument']['optical_element'], int(im.meta['instrument']['detector'][-2:])) /
             10 ** 6 / im['meta']['photometry']['pixel_area']))
        im['meta']['photometry']['conversion_megajanskys'] = val
        im['meta']['photometry']['conversion_microjanskys'] = (
            val * u.MJy / u.sr).to(u.uJy / u.arcsec ** 2).value

    im['meta']['photometry']['conversion_megajanskys_uncertainty'] = 0
    im['meta']['photometry']['conversion_microjanskys_uncertainty'] = 0


def merge_dicts(a, b):
    """Merge two dictionaries, replacing values in a with values in b.

    When both a & b have overlapping dictionaries, this recursively
    merges their contents.  If a key does not correspond to a dictionary
    in both a & b, then the content of a is overwritten with the content
    of b.

    Parameters
    ----------
    a : dict
        Dictionary to update

    b : dict
        Dictionary to use to update a.

    Returns
    -------
    a : dict
        a, mutated to contain keys from b.
    """
    for key in b:
        if key in a and isinstance(a[key], Mapping) and isinstance(b[key], Mapping):
            merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]
    return a


def calc_scale_factor(date, ra, dec):
    """Calculate velocity aberration scale factor

    The L2 orbit is just a delta on the Earth's orbit. At the moment, there is no ephemeris for
    Roman yet, the Earth's barycentric velocity is used to calculate velocity aberration. A scale
    of 1.01 is applied to the velocity to scale, approximately, to what would be Roman's velocity.

    Parameters
    ----------
    date : str or astropy.Time
        The date at which to calculate the velocity.

    ra, dec: float
        The right ascension and declination of the target (or some other
        point, such as the center of a detector) in the barycentric coordinate
        system.  The equator and equinox should be the same as the coordinate
        system for the velocity. In degrees.

    Returns
    -------
    scale_factor : float
        The velocity aberration scale factor
    """
    _, velocity = get_body_barycentric_posvel('earth', date)
    velocity = 1.01 * velocity  # Move from earth to Roman.
    xyz_velocity = velocity.xyz.to(u.km / u.s)
    scale_factor, _, _ = compute_va_effects(*xyz_velocity.value, ra, dec)

    return scale_factor
