"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from astropy import units as u
import galsim
from romanisim import util


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
    from astropy.coordinates import SkyCoord
    from galsim import CelestialCoord
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
    answers[1, 1, :] = np.array([1, 5, 2]) / np.sqrt(1 + 25 + 4)
    response = util.scalergb(testim)
    assert np.allclose(response, answers)
    response = util.scalergb(testim, lumrange=[0, 10])
    answers[1, 1, :] = np.array([1, 5, 2]) / 10
    assert np.allclose(response, answers)
    response = util.scalergb(testim, scales=[1, 2, 3], lumrange=[0, 10])
    assert np.allclose(answers / np.array([1, 2, 3])[None, None, :], response)


def test_random_points_in_cap():
    """Make sure that all the points land in the cap, and that there are the
    right number, and maybe that their radial distribution isn't crazy?"""

    from astropy.coordinates import SkyCoord
    from astropy import units as u
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


# not going to test flatten / unflatten dictionary at this point, since we only
# use this for metadata management we intend to remove.
