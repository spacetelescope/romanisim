"""
Unit tests for wcs module.
"""
import os
import copy
import numpy as np
from astropy.modeling import rotations, projections, models
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from romanisim import wcs, util, parameters
import galsim
import pytest

import roman_datamodels


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
    assert cc.separation(gwcs.pixel_to_world(0, 0)).to(u.arcsec).value < 1e-3
    cc2 = SkyCoord(ra=0 * u.deg, dec=0.11 * u.arcsec)
    assert cc2.separation(gwcs.pixel_to_world(0, 1)).to(u.arcsec).value < 1e-2
    # a tenth of a pixel
    cc3 = SkyCoord(ra=0.11 * u.arcsec, dec=0 * u.deg)
    assert cc3.separation(gwcs.pixel_to_world(1, 0)).to(u.arcsec).value < 1e-2
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

    metadata = {'instrument':
                {
                    'detector': 'WFI01',
                },
                'exposure':
                {
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
    metadata = dict(pointing=dict(), instrument=dict(), wcsinfo=dict(), velocity_aberration=dict())
    metadata['instrument']['detector'] = 'WFI01'
    util.update_pointing_and_wcsinfo_metadata(metadata, wcs.GWCS(gwcs))
    assert metadata['wcsinfo']['aperture_name'] == 'WFI01_FULL'


def test_wcs_from_fits_header():
    wcs1 = wcs.get_wcs(parameters.default_parameters_dictionary,
                       usecrds=False)
    wcs2 = wcs.wcs_from_fits_header(wcs1.header.header)
    xg, yg = np.meshgrid(np.linspace(0, 4088, 100), np.linspace(0, 4088, 100))
    rd1 = np.degrees(wcs1._radec(xg.copy(), yg.copy()))
    rd2 = wcs2.pixel_to_world(xg.copy(), yg.copy())
    rd1 = SkyCoord(ra=rd1[0] * u.deg, dec=rd1[1] * u.deg, frame=rd2.frame)
    sep = rd1.separation(rd2).to(u.arcsec).value

    # really end up with 10^-10 arcsec in my tests, but let's say that 10^-5
    # is fine.
    assert np.max(sep) < 1e-5


def test_wcs_crds_match():
    # Set up parameters for simulation run

    metadata = copy.deepcopy(parameters.default_parameters_dictionary)
    metadata['instrument']['detector'] = 'WFI07'
    metadata['instrument']['optical_element'] = 'F158'
    metadata['exposure']['ma_table_number'] = 1

    image_mod = roman_datamodels.datamodels.ImageModel.create_fake_data({"meta": metadata})
    image_mod.meta.wcs = None

    twcs = wcs.get_wcs(image_mod, usecrds=True)

    # Expect a GWCS object as opposed to a dictionary
    assert type(twcs) is wcs.GWCS


def test_scale_factor():
    """Test that specifying a scale factor actually calculates a new wcs"""

    # Truth aiming for.
    cc = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
    scale_factor = 1.1
    truth = SkyCoord([(0.        , 0.        ), (0.03361111, 0.0336111 ),
                      (0.0672222 , 0.06722216), (0.10083325, 0.10083312),
                      (0.13444424, 0.13444393)],
                     unit='deg')

    # Make a simple grid of pixel coordinates to calculate sky for
    grid = range(0, 4096, 1000)

    # Create the wcs and generate results
    distortion = make_fake_distortion_function()
    gwcs = wcs.make_wcs(cc, distortion, scale_factor=scale_factor)
    sky = gwcs.pixel_to_world(grid, grid)

    assert all(truth.separation(sky).to(u.arcsec).value < 1e-3)


def test_scale_factor_negative():
    """Test that specifying a scale factor actually calculates a new wcs"""

    # Truth aiming for.
    cc = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
    scale_factor = -99999
    truth = SkyCoord([(0.        , 0.        ), (0.03055555, 0.03055555),
                      (0.06111109, 0.06111105), (0.09166659, 0.09166647),
                      (0.12222204, 0.12222176)],
                     unit='deg')

    # Make a simple grid of pixel coordinates to calculate sky for
    grid = range(0, 4096, 1000)

    # Create the wcs and generate results
    distortion = make_fake_distortion_function()
    gwcs = wcs.make_wcs(cc, distortion, scale_factor=scale_factor)
    sky = gwcs.pixel_to_world(grid, grid)

    assert all(truth.separation(sky).to(u.arcsec).value < 1e-3)
