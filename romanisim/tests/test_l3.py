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
    metadata['basic']['optical_element'] = 'F158'

    # Create WCS
    twcs = wcs.get_mosaic_wcs(metadata, shape=(galsim.roman.n_pix, galsim.roman.n_pix))

    # Create initial Level 3 mosaic

    # Create Four-quadrant pattern of gaussian noise, centered around one
    # Each quadrant's gaussian noise scales like total exposure time
    # (total files contributed to each quadrant)

    # Create gaussian noise generators
    g1 = galsim.GaussianDeviate(rng_seed, mean=1.0, sigma=0.01)
    g2 = galsim.GaussianDeviate(rng_seed, mean=1.0, sigma=0.02)
    g3 = galsim.GaussianDeviate(rng_seed, mean=1.0, sigma=0.05)
    g4 = galsim.GaussianDeviate(rng_seed, mean=1.0, sigma=0.1)

    # Create level 3 mosaic model
    l3_mos = maker_utils.mk_level3_mosaic(shape=(galsim.roman.n_pix, galsim.roman.n_pix))

    # Update metadata in the l3 model
    for key in metadata.keys():
        if key in l3_mos.meta:
            l3_mos.meta[key].update(metadata[key])

    # Obtain unit conversion factor
    unit_factor = parameters.reference_data['photom'][l3_mos.meta.basic.optical_element]

    # Populate the mosaic data array with gaussian noise from generators
    g1.generate(l3_mos.data.value[0:100, 0:100])
    g2.generate(l3_mos.data.value[0:100, 100:200])
    g3.generate(l3_mos.data.value[100:200, 0:100])
    g4.generate(l3_mos.data.value[100:200, 100:200])
    l3_mos.data *= unit_factor.value

    # Define Poisson Noise of mosaic
    l3_mos.var_poisson.value[0:100, 0:100] = 0.01**2
    l3_mos.var_poisson.value[0:100, 100:200] = 0.02**2
    l3_mos.var_poisson.value[100:200, 0:100] = 0.05**2
    l3_mos.var_poisson.value[100:200, 100:200] = 0.1**2

    # Create normalized psf source catalog (same source in each quadrant)
    sc_dict = {"ra": 4 * [0.0], "dec": 4 * [0.0], "type": 4 * ["PSF"], "n": 4 * [-1.0],
               "half_light_radius": 4 * [0.0], "pa": 4 * [0.0], "ba": 4 * [1.0], "F158": 4 * [1.0]}
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

    source_cat = catalog.table_to_catalog(sc_table, ["F158"])

    # Copy original Mosaic before adding sources as sources are added in place
    l3_mos_orig = l3_mos.copy()
    l3_mos_orig.data = l3_mos.data.copy()
    l3_mos_orig.var_poisson = l3_mos.var_poisson.copy()

    # Add source_cat objects to mosaic
    l3.add_objects_to_l3(l3_mos, source_cat, Ct, unit_factor=unit_factor.value, seed=rng_seed)

    # Create overall scaling factor map
    Ct_all = np.divide(l3_mos_orig.data.value, l3_mos_orig.var_poisson.value,
                       out=np.ones(l3_mos_orig.data.shape), where=l3_mos_orig.var_poisson.value != 0)

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


def test_sim_mosaic():
    """Simulating mosaic from catalog file.
    """

    ra_ref = 1.0
    dec_ref = 4.0

    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)
    metadata['basic']['optical_element'] = 'F158'
    metadata['wcsinfo']['ra_ref'] = ra_ref
    metadata['wcsinfo']['dec_ref'] = dec_ref
    metadata['wcsinfo']['pixel_scale'] = 0.11
    metadata['wcsinfo']['pixel_scale_local'] = 0.11
    metadata['wcsinfo']['v2_ref'] = 0
    metadata['wcsinfo']['v3_ref'] = 0

    exptimes = [600]

    cen = SkyCoord(ra=ra_ref * u.deg, dec=dec_ref * u.deg)
    cat = catalog.make_dummy_table_catalog(cen, radius=0.01, nobj=100)
    cat['F158'] = cat['F158'] * 10e10

    source_cat = catalog.table_to_catalog(cat, ["F158"])

    filter = metadata['basic']['optical_element']

    # if context is None:
    # Create geometry from the object list
    twcs = romanisim.wcs.get_mosaic_wcs(metadata)

    coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                        for o in source_cat])

    allx, ally = twcs.radecToxy(coords[:, 0], coords[:, 1], 'rad')

    # Obtain the sample extremums
    xmin = min(allx)
    xmax = max(allx)
    ymin = min(ally)
    ymax = max(ally)

    print(f"XXX X: min, max = {xmin, xmax}")
    print(f"XXX Y: min, max = {ymin, ymax}")

    # Obtain WCS center
    xcen, ycen = twcs.radecToxy(twcs.center.ra, twcs.center.dec, 'rad')

    # Determine maximum extremums from WCS center
    xdiff = max([math.ceil(xmax - xcen), math.ceil(xcen - xmin)]) + 1
    ydiff = max([math.ceil(ymax - ycen), math.ceil(ycen - ymin)]) + 1

    print(f"XXX Test Center: xcen, ycen = {xcen}, {ycen}")
 
    print(f"XXX 2*xdiff = {2*xdiff}")
    print(f"XXX 2*ydiff = {2*ydiff}")

    # Create context map preserving WCS center
    # context = np.ones((1, 2 * xdiff, 2 * ydiff), dtype=np.uint32)
    context = np.ones((1, 2 * ydiff, 2 * xdiff), dtype=np.uint32)

    # Generate WCS
    moswcs = romanisim.wcs.get_mosaic_wcs(metadata, shape=(context.shape[1:])) #shape=context.shape[-2:])

    psf = romanisim.psf.make_psf(filter_name=filter, sca=CENTER_SCA, chromatic=False, webbpsf=True)

    # mosaic, extras = l3.simulate(metadata, source_cat, exptimes)
    # l3.simulate(shape, wcs, efftime, filter, catalog, effreadnoise=None, sky=None, psf=None):
    mosaic, extras = l3.simulate(context.shape[1:], twcs, exptimes[0], filter, source_cat, metadata=metadata)

    import plotly.express as px

    fig1 = px.imshow(mosaic.data.value, title='Mosaic Data', labels={'color': 'MJy / sr'})
    fig1.show()

    # fig2 = px.imshow(mosaic.context[-2:].reshape(mosaic.context.shape[-2:]), title='Mosaic Context', labels={'color': 'File Number'})
    # fig2.show()

    pos_vals = mosaic.data.value.copy()
    pos_vals[pos_vals <= 0] = 0.00000000001

    fig3 = px.imshow(np.log(pos_vals), title='Mosaic Data (log)', labels={'color': 'MJy / sr'})
    fig3.show()

    # mosaic, extras = l3.simulate(metadata, source_cat, exptimes)
    # l3.simulate(shape, wcs, efftime, filter, catalog, effreadnoise=None, sky=None, psf=None):
    efftimes = util.decode_context_times(context, exptimes)

    fig3 = px.imshow(mosaic.context[1:].reshape(mosaic.context.shape[1:]), title='Mosaic Context', labels={'color': 'File Number'})
    fig3.show()

    fig4 = px.imshow(efftimes, title='Effective Times', labels={'color': 'seconds'})
    fig4.show()

    tmp_image = galsim.ImageF(context.shape[1], context.shape[2], wcs=moswcs, xmin=0, ymin=0)

    print(f"XXX context.shape = {context.shape}")
    print(f"XXX efftimes.shape = {efftimes.shape}")
    # print(f"XXX moswcs = {moswcs}")
    # print(f"XXX twcs = {twcs}")
    # print(f"XXX tmp_image.wcs = {tmp_image.wcs}")

    xmoscen, ymoscen = moswcs.radecToxy(moswcs.center.ra, moswcs.center.dec, 'rad')
    print(f"XXX Test Center: xmoscen, ymoscen = {xmoscen}, {ymoscen}")

    mosaic2, extras2 = l3.simulate(context.shape[1:], tmp_image.wcs, efftimes, filter, source_cat, metadata=metadata)

    fig1 = px.imshow(mosaic2.data.value, title='Mosaic Data', labels={'color': 'MJy / sr'})
    fig1.show()



    pos_vals2 = mosaic2.data.value.copy()
    pos_vals2[pos_vals2 <= 0] = 0.00000000001

    fig3 = px.imshow(np.log(pos_vals2), title='Mosaic Data (log)', labels={'color': 'MJy / sr'})
    fig3.show()



# TBD: Test with more complex context

# TBD: Test of geometry construction
