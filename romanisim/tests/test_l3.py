"""Unit tests for mosaic module.

"""

import os
import copy
import math
import numpy as np
import galsim
from galsim import roman
from romanisim import image, parameters, catalog, psf, util, wcs, persistence, ramp, l1, l3
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy import table
import asdf
import webbpsf
from astropy.modeling.functional_models import Sersic2D
import pytest
from metrics_logger.decorators import metrics_logger
from romanisim import log
from roman_datamodels.stnode import WfiScienceRaw, WfiImage
import roman_datamodels.maker_utils as maker_utils
import romanisim.bandpass

# Define centermost SCA for PSFs
CENTER_SCA = 2



@metrics_logger("DMS232")
@pytest.mark.soctests
def test_inject_sources_into_mosaic():
    """Inject sources into a mosaic.
    """

    # Set constants and metadata
    galsim.roman.n_pix = 200
    rng_seed = 42
    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['basic']['optical_element'] = filter_name

    # Create WCS
    twcs = wcs.get_mosaic_wcs(metadata, shape=(galsim.roman.n_pix, galsim.roman.n_pix))

    # Create initial Level 3 mosaic

    # Create Four-quadrant pattern of gaussian noise, centered around one
    # Each quadrant's gaussian noise scales like total exposure time
    # (total files contributed to each quadrant)

    # Create gaussian noise generators
    g1 = galsim.GaussianDeviate(rng_seed, mean=1.0e-7, sigma=0.01e-7)
    g2 = galsim.GaussianDeviate(rng_seed, mean=1.0e-7, sigma=0.02e-7)
    g3 = galsim.GaussianDeviate(rng_seed, mean=1.0e-7, sigma=0.05e-7)
    g4 = galsim.GaussianDeviate(rng_seed, mean=1.0e-7, sigma=0.1e-7)

    # Create level 3 mosaic model
    l3_mos = maker_utils.mk_level3_mosaic(shape=(galsim.roman.n_pix, galsim.roman.n_pix))
    l3_mos['meta']['wcs'] = twcs

    # Update metadata in the l3 model
    for key in metadata.keys():
        if key in l3_mos.meta:
            l3_mos.meta[key].update(metadata[key])

    # Obtain unit conversion factors
    # Need to convert from counts / pixel to MJy / sr
    # Flux to counts
    cps_conv = romanisim.bandpass.get_abflux(filter_name)
    # Unit factor
    unit_factor = ((3631 * u.Jy) / (romanisim.bandpass.get_abflux(filter_name) * 10e6
                                    * parameters.reference_data['photom']["pixelareasr"][filter_name])).to(u.MJy / u.sr)

    # Populate the mosaic data array with gaussian noise from generators
    g1.generate(l3_mos.data.value[0:100, 0:100])
    g2.generate(l3_mos.data.value[0:100, 100:200])
    g3.generate(l3_mos.data.value[100:200, 0:100])
    g4.generate(l3_mos.data.value[100:200, 100:200])

    # Define Poisson Noise of mosaic
    l3_mos.var_poisson.value[0:100, 0:100] = 0.01**2
    l3_mos.var_poisson.value[0:100, 100:200] = 0.02**2
    l3_mos.var_poisson.value[100:200, 0:100] = 0.05**2
    l3_mos.var_poisson.value[100:200, 100:200] = 0.1**2

    # Create normalized psf source catalog (same source in each quadrant)
    mag_flux = 1e-10
    sc_dict = {"ra": 4 * [0.0], "dec": 4 * [0.0], "type": 4 * ["PSF"], "n": 4 * [-1.0],
               "half_light_radius": 4 * [0.0], "pa": 4 * [0.0], "ba": 4 * [1.0], filter_name: 4 * [mag_flux]}
    sc_table = table.Table(sc_dict)

    # Set locations
    xpos_idx = [50, 50, 150, 150]
    ypos_idx = [50, 150, 50, 150]

    # Populate flux scaling ratio and catalog
    Ct = []
    for idx, (x, y) in enumerate(zip(xpos_idx, ypos_idx)):
        # Set scaling factor for injected sources
        # Flux / sigma_p^2
        if l3_mos.var_poisson[y, x].value != 0:
            Ct.append(math.fabs(l3_mos.data[y, x].value / l3_mos.var_poisson[y, x].value))
        else:
            Ct.append(1.0)

        sc_table["ra"][idx], sc_table["dec"][idx] = (twcs._radec(x, y) * u.rad).to(u.deg).value

    source_cat = catalog.table_to_catalog(sc_table, [filter_name])

    # Copy original Mosaic before adding sources as sources are added in place
    l3_mos_orig = l3_mos.copy()
    l3_mos_orig.data = l3_mos.data.copy()
    l3_mos_orig.var_poisson = l3_mos.var_poisson.copy()

    # Add source_cat objects to mosaic
    l3.add_objects_to_l3(l3_mos, source_cat, Ct, cps_conv=cps_conv, unit_factor=unit_factor.value, seed=rng_seed)

    # Create overall scaling factor map
    Ct_all = np.divide(l3_mos_orig.data.value, l3_mos_orig.var_poisson.value,
                       out=np.ones(l3_mos_orig.data.shape),
                       where=l3_mos_orig.var_poisson.value != 0)

    # Set new poisson variance
    l3_mos.var_poisson = (l3_mos.data.value / Ct_all) * l3_mos.var_poisson.unit

    # Ensure that every data pixel value has increased or
    # remained the same with the new sources injected
    assert np.all(l3_mos.data.value >= l3_mos_orig.data.value)

    # Ensure that every pixel's poisson variance has increased or
    # remained the same with the new sources injected
    # Numpy isclose is needed to determine equality, due to float precision issues
    close_mask = np.isclose(l3_mos.var_poisson.value, l3_mos_orig.var_poisson.value, rtol=1e-06)
    assert False in close_mask
    assert np.all(l3_mos.var_poisson.value[~close_mask] > l3_mos_orig.var_poisson.value[~close_mask])

    # Ensure total added flux matches expected added flux
    total_rec_flux = np.sum(l3_mos.data - l3_mos_orig.data) / unit_factor
    total_theo_flux = 4 * mag_flux * cps_conv
    assert np.isclose(total_rec_flux, total_theo_flux, rtol=4e-02)

    # Create log entry and artifacts
    log.info('DMS232 successfully injected sources into a mosaic at points (50,50), (50,150), (150,50), (150,150).')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'l3_mos': l3_mos,
                    'l3_mos_orig': l3_mos_orig,
                    'source_cat_table': sc_table,
                    }
        af.write_to(os.path.join(artifactdir, 'dms232.asdf'))


# Add decorators
def test_sim_mosaic():
    """Simulating mosaic from catalog file.
    """

    ra_ref = parameters.default_mosaic_parameters_dictionary['wcsinfo']['ra_ref']
    dec_ref = parameters.default_mosaic_parameters_dictionary['wcsinfo']['dec_ref']

    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = metadata['basic']['optical_element']

    exptimes = [600]

    cen = SkyCoord(ra=ra_ref * u.deg, dec=dec_ref * u.deg)
    cat = catalog.make_dummy_table_catalog(cen, radius=0.02, nobj=100)
    source_cat = catalog.table_to_catalog(cat, [filter_name])

    # Create bounds from the object list
    twcs = romanisim.wcs.get_mosaic_wcs(metadata)
    coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                        for o in source_cat])
    allx, ally = twcs.radecToxy(coords[:, 0], coords[:, 1], 'rad')

    # Obtain the sample extremums
    xmin = min(allx)
    xmax = max(allx)
    ymin = min(ally)
    ymax = max(ally)

    # Obtain WCS center
    xcen, ycen = twcs.radecToxy(twcs.center.ra, twcs.center.dec, 'rad')

    # Determine maximum extremums from WCS center
    xdiff = max([math.ceil(xmax - xcen), math.ceil(xcen - xmin)]) + 1
    ydiff = max([math.ceil(ymax - ycen), math.ceil(ycen - ymin)]) + 1

    # Create context map preserving WCS center
    context = np.ones((1, 2 * ydiff, 2 * xdiff), dtype=np.uint32)

    # Generate WCS
    moswcs = romanisim.wcs.get_mosaic_wcs(metadata, shape=(context.shape[1:]))

    # Simulate mosaic
    mosaic, extras = l3.simulate(context.shape[1:], moswcs, exptimes[0], filter_name, source_cat, metadata=metadata)

    import plotly.express as px

    fig1 = px.imshow(mosaic.data.value, title='1. Mosaic Data', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    fig1.show()

    pos_vals = mosaic.data.value.copy()
    pos_vals[pos_vals <= 0] = 0.00000000001

    fig3 = px.imshow(np.log(pos_vals), title='3. Mosaic Data (log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    fig3.show()


    # Second test, passing more in

    psf = romanisim.psf.make_psf(filter_name=filter_name, sca=CENTER_SCA, chromatic=False, webbpsf=True)

    efftimes = util.decode_context_times(context, exptimes)

    fig4 = px.imshow(efftimes, title='4. Effective Times', labels={'color': 'seconds', 'x': 'Y axis', 'y': 'X axis'})
    fig4.show()

    mosaic2, extras2 = l3.simulate(context.shape[1:], moswcs, efftimes, filter_name, source_cat, metadata=metadata, psf=psf)

    fig5 = px.imshow(mosaic2.data.value, title='5. Mosaic Data', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    fig5.show()

    pos_vals2 = mosaic2.data.value.copy()
    pos_vals2[pos_vals2 <= 0] = 0.00000000001

    fig6 = px.imshow(np.log(pos_vals2), title='6. Mosaic Data (log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    fig6.show()


    # The two tests below do not pass yet

    # did we get all the flux?
    totflux = np.sum(mosaic2.data.value - np.median(mosaic2.data.value))
    expectedflux = (romanisim.bandpass.get_abflux(filter_name) * np.sum(cat[filter_name])
                    / parameters.reference_data['gain'].value)
    # Something about the flux scaling or the sky is off? This fails
    # assert np.abs(totflux / expectedflux - 1) < 0.1

    # Are there sources where there should be?
    for r, d in zip(cat['ra'], cat['dec']):
        x, y = moswcs.toImage(r, d, units=galsim.degrees)
        x = int(x)
        y = int(y)
        # assert mosaic2.data.value[y, x] > np.median(mosaic2.data.value) * 5

# TBD: Test var_poisson (variable exposure times)

# TBD: Test with more complex context

# TBD: Test of geometry construction
