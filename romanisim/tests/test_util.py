"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time
import galsim
from romanisim import util
from astropy.coordinates import SkyCoord
from galsim import CelestialCoord


def test_dummy():
    assert 1 > 0


@pytest.mark.soctests
def test_dummy_soctest():
    assert 1 > 0


def test_coordconv():
    """Make some random points, do some conversions, make sure that all of
    the values are close to where they started."""
    npts = 100
    ra = np.random.uniform(-720, 720, npts)
    dec = np.random.uniform(-90, 90, npts)
    skycoord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    celcoord = [CelestialCoord(r * galsim.degrees, d * galsim.degrees)
                for (r, d) in zip(ra, dec)]
    celcoordp = [util.skycoord(c) for c in celcoord]
    skycoordp = [util.celestialcoord(s) for s in skycoord]
    eps = 1.e-6
    for i, (c, s) in enumerate(zip(celcoordp, skycoordp)):
        c1 = c
        c2 = SkyCoord(s.ra / galsim.radians * u.rad,
                      s.dec / galsim.radians * u.rad)
        c3 = SkyCoord(ra[i] * u.deg, dec[i] * u.deg)
        v1 = c1.separation(c2).to(u.deg).value
        v2 = c1.separation(c3).to(u.deg).value
        assert v1 < eps
        assert v2 < eps
    assert util.skycoord(skycoord) is skycoord
    assert util.celestialcoord(celcoord[0]) is celcoord[0]


def test_scalergb():
    """Put together a really annoying image and make sure the values are all
    what is expected.
    """
    testim = np.zeros((2, 2, 3), dtype='f4')
    answers = np.zeros((2, 2, 3), dtype='f4')
    testim[0, 0, :] = [0, 0, 0]
    answers[0, 0, :] = 0
    testim[0, 1, :] = [-1, -1, -1]
    answers[0, 1, :] = 0
    testim[1, 0, :] = [-1000, 0, 0]
    answers[1, 0, :] = 0
    testim[1, 1, :] = [1, 5, 2]
    answers[1, 1, :] = np.array([1, 5, 2]) * np.arcsinh(20 * 8) / np.sqrt(20) / 8
    response = util.scalergb(testim)
    assert np.allclose(response, answers)
    response = util.scalergb(testim, scales=[1, 2, 3])
    newanswers = util.scalergb(testim / np.array([1, 2, 3])[None, None, :])
    assert np.allclose(response, newanswers)


def test_random_points_in_cap():
    """Make sure that all the points land in the cap, and that there are the
    right number, and maybe that their radial distribution isn't crazy?"""

    npts = 10000
    cen = SkyCoord(ra=60 * u.deg, dec=-10 * u.deg)
    rad = 5
    pts = util.random_points_in_cap(cen, rad, npts)
    assert len(pts) == npts
    seps = cen.separation(pts).to(u.deg).value
    assert np.max(seps) < 5
    fracininnerhalf = np.sum(seps < 5 / np.sqrt(2)) / npts
    # should be distributed with stdev ~ sqrt(0.5*npts)/npts = sqrt(0.5/npts)
    assert np.abs(fracininnerhalf - 0.5) < 10 * np.sqrt(0.5 / npts)


def test_add_more_metadata():
    metadata = {'exposure':
                {
                    'start_time': Time('2026-01-01T00:00:00'),
                    'ma_table_number': 4,
                },
                'instrument':
                {
                    'detector': 'WFI01',
                },
                }
    util.add_more_metadata(metadata)
    assert len(metadata['exposure']) > 2  # some metadata got added.


def test_king_profile():
    """Test King (1962) profile routines."""
    # king_profile, sample_king_distances, random_points_in_king

    # truncation radius
    assert util.king_profile(2, 1, 2) == 0
    # decreasing
    xx = np.linspace(0, 2, 1000)
    profile = util.king_profile(xx, 1, 2)
    assert np.all(np.diff(profile) < 0)

    xx = util.sample_king_distances(1, 2, 1000)
    assert np.max(xx) < 2
    # nothing beyond the truncation radius

    cen = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
    cc = util.random_points_in_king(cen, 1, 2, 1000)
    sep = cen.separation(cc).to(u.deg).value
    assert np.max(sep) < 2
    # truncation radius


def test_random_points_at_radii():
    """Test offseting points by random angles and given distances."""

    n = 1000
    coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
    distances = np.random.uniform(0, 1, n) * u.deg
    newpoints = util.random_points_at_radii(coord, distances)
    sep = coord.separation(newpoints)
    assert np.allclose(sep, distances)
    posangle = coord.position_angle(newpoints)
    # these should be roughly uniform over 2pi.
    # What's the right test for that?
    from astropy.stats import circstd
    assert circstd(posangle, method='angular') > 1
    # I ran 10k runs of this; peaks at 1.4 with a standard deviation
    # of ~0.02, smallest value was 1.34.  Should never be
    # less than 1, or even 1.3.


def test_merge_dicts():
    res = util.merge_dicts(dict(), dict(a='a', b='b'))
    assert len(res) == 2
    res = util.merge_dicts(dict(a=dict(a1='hello')),
                           dict(a=dict(a1='goodbye')))
    assert (len(res) == 1) and (res['a']['a1'] == 'goodbye')
    res = util.merge_dicts(dict(a=dict(a1='hello')),
                           dict(a='hello'))
    assert (len(res) == 1) and (res['a'] == 'hello')
