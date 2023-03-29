"""
Unit tests for wcs module.
"""

import numpy as np
from astropy.modeling import rotations, projections, models
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from romanisim import wcs, util, parameters
import galsim


def make_fake_distortion_function():
    """A very simple distortion function.

    This returns a very simple distortion function that scales pixels
    by roughly their angular size on the sky and does a tangent plane
    projection.  To make this more right we'd have to start by looking
    at where detectors actually are relative to the boresight and put in
    a more complicated radial distortion function than just tangent plane.
    One could imagine that going into pix2tan.  But I don't think that
    actually improves our tests at present so I'm leaving this as a
    vaguely plausible ("does kind of the right thing") distortion function.
    """
    # distortion takes from pixels to V2 V3
    # V2V3 is a weird spherical coordinate system where 0, 0 -> the boresight.
    zen2v2v3 = (rotations.EulerAngleRotation(0, -90, 0, 'xyz')
                | (models.Scale(3600) & models.Scale(3600)))
    # zenith -> (0, 0) rotation
    tanproj = projections.Pix2Sky_Gnomonic()
    pix2tan = models.Scale(0.11 / 3600) & models.Scale(0.11 / 3600)
    return pix2tan | tanproj | zen2v2v3


def test_wcs():
    distortion = make_fake_distortion_function()
    cc = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
    gwcs = wcs.make_wcs(cc, distortion)
    assert cc.separation(gwcs(0, 0, with_units=True)).to(u.arcsec).value < 1e-3
    cc2 = SkyCoord(ra=0 * u.deg, dec=0.11 * u.arcsec)
    assert cc2.separation(gwcs(0, 1, with_units=True)).to(u.arcsec).value < 1e-2
    # a tenth of a pixel
    cc3 = SkyCoord(ra=0.11 * u.arcsec, dec=0 * u.deg)
    assert cc3.separation(gwcs(1, 0, with_units=True)).to(u.arcsec).value < 1e-2
    wcsgalsim = wcs.GWCS(gwcs)
    assert wcsgalsim.wcs is gwcs
    assert ((wcsgalsim.origin.x == 0) and (wcsgalsim.origin.y == 0))
    assert repr(wcsgalsim)[:10] == 'romanisim.'
    wcsgalsim2 = wcsgalsim.copy()
    pos = galsim.PositionD(128, 256)
    cc1 = wcsgalsim.toWorld(pos)
    cc2 = wcsgalsim2.toWorld(pos)
    assert np.allclose([cc1.ra / galsim.degrees, cc1.dec / galsim.degrees],
                       [cc2.ra / galsim.degrees, cc2.dec / galsim.degrees])
    pos2 = wcsgalsim.toImage(cc1)
    assert np.allclose([pos.x, pos.y], [pos2.x, pos2.y])
    # also try some arrays
    xx = np.random.uniform(0, 4096, 100)
    yy = np.random.uniform(0, 4096, 100)
    rr, dd = wcsgalsim._radec(xx, yy)
    xx2, yy2 = wcsgalsim._xy(rr, dd)
    assert np.allclose(xx, xx2)
    assert np.allclose(yy, yy2)
    # metadata = {'roman.meta.instrument.detector': 'WF101',
    #             'roman.meta.exposure.start_time': Time('2026-01-01T00:00:00')
    #             }
    metadata = {'instrument' : {
                    'detector': 'WF101',
                },
                'exposure' : {
                    'start_time': Time('2026-01-01T00:00:00'),
                }
    }
    wcs.fill_in_parameters(metadata, cc, boresight=True)
    wcswrap = wcs.get_wcs(metadata, distortion=distortion)
    cc2 = util.skycoord(wcswrap.toWorld(galsim.PositionD(0, 0)))
    assert cc.separation(cc2).to(u.arcsec).value < 1e-3
    celpole = SkyCoord(270 * u.deg, 66 * u.deg)
    wcs.fill_in_parameters(metadata, celpole, boresight=True)
    wcsgalsim = wcs.get_wcs(metadata, usecrds=False)
    cc3 = util.skycoord(wcsgalsim.toWorld(galsim.PositionD(128, 128)))
    assert cc3.separation(celpole).to(u.degree).value < 3
    # big margin here of 3 deg!  The CCDs are not really at the boresight.
    wcs.fill_in_parameters(metadata, cc, boresight=True)
    wcswrap2 = wcs.get_wcs(metadata, distortion=distortion)
    cc3 = util.skycoord(wcswrap2.toWorld(galsim.PositionD(0, 0)))
    assert cc.separation(cc3).to(u.arcsec).value < 1e-3
    wcs.fill_in_parameters(metadata, cc, boresight=False)
    wcswrap3 = wcs.get_wcs(metadata, distortion=distortion)
    cc4 = util.skycoord(wcswrap3.toWorld(galsim.PositionD(0, 0)))
    # The difference between locations with and without the boresight offset
    # should be close to the reference v2 & v3 offset.
    assert np.abs(cc3.separation(cc4).to(u.arcsec).value
                  - np.hypot(*parameters.v2v3_wficen)) < 1
