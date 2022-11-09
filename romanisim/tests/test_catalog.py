"""
Unit tests for catalog functions.
"""

import os
import urllib.request
import numpy as np
import galsim
from romanisim import catalog
from astropy.coordinates import SkyCoord
from astropy import units as u

cosmos_url = ('https://github.com/GalSim-developers/GalSim/raw/releases/'
              '2.4/examples/data/real_galaxy_catalog_23.5_example.fits')

def test_make_dummy_catalog(tmp_path):
    cen = SkyCoord(ra=5 * u.deg, dec=-10 * u.deg)
    radius = 0.2
    nobj = 100
    fn = os.path.join(tmp_path, 'example.fits')
    urllib.request.urlretrieve(cosmos_url, fn)
    urllib.request.urlretrieve(cosmos_url.replace('.fits', '_selection.fits'),
                               fn.replace('.fits', '_selection.fits'))
    urllib.request.urlretrieve(cosmos_url.replace('.fits', '_fits.fits'),
                               fn.replace('.fits', '_fits.fits'))
    from astropy.io import fits
    print(fits.getdata(fn).dtype)
    cat = catalog.make_dummy_catalog(
        cen, radius=radius, seed=11, nobj=nobj, chromatic=True,
        galaxy_sample_file_name=str(fn))
    assert len(cat) == nobj
    skycoord = SkyCoord(
        ra=[c.sky_pos.ra / galsim.degrees * u.deg for c in cat],
        dec=[c.sky_pos.dec / galsim.degrees * u.deg for c in cat])
    assert np.max(cen.separation(skycoord).to(u.deg).value) < radius
    assert cat[0].profile.spectral
    assert cat[0].flux is None  # fluxes built into profile
    cat = catalog.make_dummy_catalog(
        cen, radius=radius, nobj=nobj, chromatic=False,
        galaxy_sample_file_name=str(fn))
    assert not cat[0].profile.spectral


def test_table_catalog(tmp_path):
    cen = SkyCoord(ra=5 * u.deg, dec=-10 * u.deg)
    radius = 0.2
    nobj = 200
    bands = ['F087', 'hello']
    table = catalog.make_dummy_table_catalog(cen, radius=radius, nobj=nobj,
                                             bandpasses=bands)
    coord = SkyCoord(ra=table['ra'] * u.deg, dec=table['dec'] * u.deg)
    assert len(table) == nobj
    assert np.max(cen.separation(coord).to(u.deg).value) < radius
    assert np.sum(table['type'] == 'PSF') != 0
    assert np.sum(table['type'] == 'SER') != 0
    m = table['type'] == 'PSF'
    assert np.all(table['n'][m] == -1)
    assert np.all(table['n'][~m] != -1)
    assert np.all(table['half_light_radius'][m] == 0)
    assert np.all(table['half_light_radius'][~m] != 0)
    assert np.all(table['pa'][m] == 0)
    assert np.all(table['ba'][m] == 1)
    assert np.all(table['pa'][~m] != 0)
    assert np.all(table['ba'][~m] != 0)
    print(table.dtype.names)
    for b in bands:
        assert b in table.dtype.names
    cat = catalog.table_to_catalog(table, bands)
    assert not cat[0].profile.spectral  # table-based profiles are not chromatic
    assert len(cat[0].flux) == len(bands)  # has some fluxes
    assert cat[0].flux[bands[0]] == table[bands[0]][0]

    tabpath = tmp_path / 'table.ecsv'
    table.write(tabpath)
    newcat = catalog.read_catalog(tabpath, bands)
    assert len(newcat) == len(cat)
    for c1, c2 in zip(cat, newcat):
        assert c1.sky_pos == c2.sky_pos
        assert c1.profile == c2.profile
        assert np.all([c1.flux[b] == c2.flux[b] for b in bands])

    table = catalog.make_dummy_table_catalog(cen, radius=radius, nobj=nobj)
    assert ('F087' in table.dtype.names) or ('Z087' in table.dtype.names)
