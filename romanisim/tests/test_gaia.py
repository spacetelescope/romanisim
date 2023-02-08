import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from romanisim import gaia


def test_gaia():
    fakegaiacat = Table()
    fakegaiacat['ra'] = np.array([270.0] * 5) * u.deg
    fakegaiacat['dec'] = np.array([66.0, 66.0, -24.0, 0, 0]) * u.deg
    fakegaiacat['pmra'] = np.array([1, 0, 0, 1, 0]) * u.mas / u.year
    fakegaiacat['pmdec'] = np.array([1, 0, 0, 1, 0]) * u.mas / u.year
    fakegaiacat['parallax'] = np.array([0, 1, 1, 0, 0]) * u.mas
    fakegaiacat['phot_g_mean_mag'] = 15.0
    dates = Time(
        np.linspace(2016, 2017, 10), format='jyear')
    cats = [gaia.gaia2romanisimcat(
        fakegaiacat, date, refepoch=2016, boost_parallax=1) for date in dates]

    coords = SkyCoord(ra=np.array([c['ra'] for c in cats]) * u.deg,
                      dec=np.array([c['dec'] for c in cats]) * u.deg)
    origcoord = SkyCoord(fakegaiacat['ra'], fakegaiacat['dec'])
    sep = coords.separation(origcoord)
    maxsep = np.max(sep, axis=0)

    for field in ['ra', 'dec']:
        # first source doesn't move it first catalog since it's observed at its
        # epoch and has zero parallax
        assert cats[0][field][0] == fakegaiacat[field][0]
    # max separation when pm = 0 should be roughly parallax / 2
    pm0 = (fakegaiacat['pmra'] == 0) & (fakegaiacat['pmdec'] == 0)
    assert np.all(np.abs(maxsep[pm0] - fakegaiacat['parallax'][pm0] / 2)
                  <= fakegaiacat['parallax'][pm0] * 0.01)
    # sources with zero pmra and plx never move
    assert np.all(sep[:, -1] == 0 * u.deg)
    # sources with zero parallax have increasing separation
    assert np.all(np.diff(sep[:, 0]) > 0)
    assert np.all(np.diff(sep[:, 3]) > 0)
