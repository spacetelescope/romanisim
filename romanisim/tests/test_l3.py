"""Unit tests for mosaic module.

"""

import os
import copy
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
    sc_dict = {"ra": 4 * [0.0], "dec": 4 * [0.0], "type": 4 * ["PSF"], "n": 4 * [-1.0],
               "half_light_radius": 4 * [0.0], "pa": 4 * [0.0], "ba": 4 * [1.0], "F158": 4 * [1.0]}
    sc_table = table.Table(sc_dict)

    xpos, ypos = 50, 50
    sc_table["ra"][0], sc_table["dec"][0] = (twcs._radec(xpos, ypos) * u.rad).to(u.deg).value
    xpos, ypos = 50, 150
    sc_table['ra'][1], sc_table['dec'][1] = (twcs._radec(xpos, ypos) * u.rad).to(u.deg).value
    xpos, ypos = 150, 50
    sc_table['ra'][2], sc_table['dec'][2] = (twcs._radec(xpos, ypos) * u.rad).to(u.deg).value
    xpos, ypos = 150, 150
    sc_table['ra'][3], sc_table['dec'][3] = (twcs._radec(xpos, ypos) * u.rad).to(u.deg).value

    source_cat = catalog.table_to_catalog(sc_table, ["F158"])
    coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                           for o in source_cat])

    # Copy original Mosaic before adding sources
    l3_mos_orig = l3_mos.copy()
    l3_mos_orig.data = l3_mos.data.copy()
    l3_mos_orig.var_poisson = l3_mos.var_poisson.copy()

    # Add source_cat objects to mosaic
    l3.add_objects_to_l3(l3_mos, source_cat, seed=rng_seed)
    # l3.add_objects_to_l3(l3_mos, source_cat, coords=coords, seed=rng_seed)

    import plotly.express as px

    fig1 = px.imshow(l3_mos_orig.data.value, title='Orig Mosaic Data', labels={'color': 'MJy / sr'})
    fig1.show()

    fig2 = px.imshow(l3_mos.data.value, title='Injected Mosaic Data', labels={'color': 'MJy / sr'})
    fig2.show()

    fig3 = px.imshow((l3_mos.data.value - l3_mos_orig.data.value), title='Diff Mosaic Data', labels={'color': 'MJy / sr'})
    fig3.show()


    # Ensure that every data pixel value has increased or
    # remained the same with the new sources injected
    assert np.all(l3_mos.data.value >= l3_mos_orig.data.value)

    # Ensure that every pixel's poisson variance has increased or
    # remained the same with the new sources injected
    # Numpy isclose is needed to determine equality, due to float precision issues
    close_mask = np.isclose(l3_mos.var_poisson.value, l3_mos_orig.var_poisson.value, rtol=1e-06)

    fig4 = px.imshow(close_mask.astype(int), title='Makes', labels={'color': 'T/F'})
    fig4.show()

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
