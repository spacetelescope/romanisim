"""
Unit tests for catalog functions.
"""

import os
import numpy as np
import galsim
from romanisim import catalog
from astropy.coordinates import SkyCoord
from astropy import units as u
from romanisim import log


def test_make_dummy_catalog():
    cen = SkyCoord(ra=5 * u.deg, dec=-10 * u.deg)
    radius = 0.2
    nobj = 100
    fn = os.environ.get('GALSIM_CAT_PATH', None)
    if fn is not None:
        fn = str(fn)
    cat = catalog.make_dummy_catalog(
        cen, radius=radius, seed=11, nobj=nobj, chromatic=True,
        galaxy_sample_file_name=fn)
    assert len(cat) == nobj
    skycoord = SkyCoord(
        ra=[c.sky_pos.ra / galsim.degrees * u.deg for c in cat],
        dec=[c.sky_pos.dec / galsim.degrees * u.deg for c in cat])
    assert np.max(cen.separation(skycoord).to(u.deg).value) < radius
    assert cat[0].profile.spectral
    assert cat[0].flux is None  # fluxes built into profile
    cat = catalog.make_dummy_catalog(
        cen, radius=radius, nobj=nobj, chromatic=False,
        galaxy_sample_file_name=fn)
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

    faintmag = 30
    n = 1000
    tstar = catalog.make_stars(
        cen, n=n, radius=1, index=1, faintmag=faintmag,
        truncation_radius=None, bandpasses=bands)
    assert len(tstar) == n
    assert np.all(tstar['type'] == 'PSF')
    assert np.max(-2.5 * np.log10(tstar[bands[0]])) < faintmag + 6
    assert np.max(-2.5 * np.log10(tstar[bands[0]])) > faintmag
    # sigma of 1 mag dispersion added into mags, so it's okay if we go fainter
    # than faintmag
    # some test of index or truncation radius?
    # large index -> more faint stars
    tstar2 = catalog.make_stars(
        cen, n=n, radius=1, index=10, faintmag=faintmag,
        truncation_radius=None, bandpasses=bands)
    assert np.median(tstar2[bands[0]]) < np.median(tstar[bands[0]])
    # median star gets fainter when a large index is used
    # i.e., more stars are close to the faint limit
    tstar3 = catalog.make_stars(
        cen, n=n, radius=1, index=1, faintmag=faintmag, truncation_radius=1,
        bandpasses=bands)
    tstar4 = catalog.make_stars(
        cen, n=n, radius=1, index=1, faintmag=faintmag, truncation_radius=2,
        bandpasses=bands)
    cc3 = SkyCoord(ra=tstar3['ra'] * u.deg, dec=tstar3['dec'] * u.deg)
    cc4 = SkyCoord(ra=tstar4['ra'] * u.deg, dec=tstar4['dec'] * u.deg)
    maxsep3 = np.max(cc3.separation(cen).to(u.deg).value)
    maxsep4 = np.max(cc4.separation(cen).to(u.deg).value)
    assert maxsep3 < 1
    assert maxsep4 < 2
    assert maxsep4 > maxsep3

    tgal1 = catalog.make_galaxies(
        cen, n=n, radius=1, index=1, faintmag=faintmag,
        bandpasses=bands, hlr_at_faintmag=1)
    tgal2 = catalog.make_galaxies(
        cen, n=n, radius=1, index=10, faintmag=faintmag,
        bandpasses=bands)
    tgal3 = catalog.make_galaxies(
        cen, n=n, radius=1, index=1, faintmag=faintmag,
        bandpasses=bands, hlr_at_faintmag=100)
    assert np.median(tgal2[bands[0]]) < np.median(tgal1[bands[0]])
    assert np.all(tgal1['type'] == 'SER')
    assert np.all((tgal1['pa'] > -360) & (tgal1['pa'] < 360))
    assert np.all((tgal1['ba'] > 0) & (tgal1['ba'] <= 1))
    assert np.all((tgal1['n'] > 0) & (tgal1['n'] <= 8))
    # not insane Sersic indices
    assert np.all((tgal1['half_light_radius'] > 0)
                  & (tgal1['half_light_radius'] < 3600))
    assert (np.median(tgal3['half_light_radius'])
            > np.median(tgal1['half_light_radius']))
    assert (np.median(tgal1[bands[0]])
            > np.median(tgal2[bands[0]]))
    log.info('DMS217: Generated parametric distributions of sources with '
             'different magnitudes and sizes.')
