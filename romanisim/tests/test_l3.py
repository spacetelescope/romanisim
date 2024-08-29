"""Unit tests for mosaic module.

"""

import os
import copy
import math
import numpy as np
import galsim
from romanisim import parameters, catalog, wcs, l3, psf, util
from astropy import units as u
from astropy import table
import asdf
import pytest
from metrics_logger.decorators import metrics_logger
from romanisim import log
import roman_datamodels.maker_utils as maker_utils
import romanisim.bandpass
from galsim import roman
from astropy.coordinates import SkyCoord

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


def set_up_image_rendering_things():
    im = galsim.ImageF(100, 100, scale=0.11, xmin=0, ymin=0)
    filter_name = 'F158'
    impsfgray = psf.make_psf(1, filter_name, webbpsf=True, chromatic=False,
                             nlambda=1)  # nlambda = 1 speeds tests
    impsfchromatic = psf.make_psf(1, filter_name, webbpsf=False,
                                  chromatic=True)
    bandpass = roman.getBandpasses(AB_zeropoint=True)['H158']
    counts = 1000
    fluxdict = {filter_name: counts}
    from copy import deepcopy
    graycatalog = [
        catalog.CatalogObject(None, galsim.DeltaFunction(), deepcopy(fluxdict)),
        catalog.CatalogObject(None, galsim.Sersic(1, half_light_radius=0.2),
                              deepcopy(fluxdict))
    ]
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
    vega_sed = vega_sed.withFlux(counts, bandpass)
    chromcatalog = [
        catalog.CatalogObject(None, galsim.DeltaFunction() * vega_sed, None),
        catalog.CatalogObject(
            None, galsim.Sersic(1, half_light_radius=0.2) * vega_sed, None)
    ]
    tabcat = table.Table()
    tabcat['ra'] = [270.0]
    tabcat['dec'] = [66.0]
    tabcat[filter_name] = counts
    tabcat['type'] = 'PSF'
    tabcat['n'] = -1
    tabcat['half_light_radius'] = -1
    tabcat['pa'] = -1
    tabcat['ba'] = -1
    return dict(im=im, impsfgray=impsfgray,
                impsfchromatic=impsfchromatic,
                bandpass=bandpass, counts=counts, fluxdict=fluxdict,
                graycatalog=graycatalog,
                chromcatalog=chromcatalog, filter_name=filter_name,
                tabcatalog=tabcat)


# Add decorators
def test_sim_mosaic():
    """Simulating mosaic from catalog file.
    """
    rng_seed = 42

    ra_ref = parameters.default_mosaic_parameters_dictionary['wcsinfo']['ra_ref']
    dec_ref = parameters.default_mosaic_parameters_dictionary['wcsinfo']['dec_ref']

    print(f"XXX ra_ref, dec_ref = {ra_ref}, {dec_ref}")

    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = metadata['basic']['optical_element']
    print(f"XXX filter_name = {filter_name}")

    exptimes = [600]

    cen = SkyCoord(ra=(ra_ref * u.deg).to(u.rad), dec=(dec_ref * u.deg).to(u.rad))
    # cat = catalog.make_dummy_table_catalog(cen, radius=0.02, nobj=100)
    cat = catalog.make_dummy_table_catalog(cen, radius=0.02, nobj=10, seed=rng_seed)
    cat[filter_name] *= 1e4
    source_cat = catalog.table_to_catalog(cat, [filter_name])

    print(f"XXX {10**(-2.5 * 25.42086391)} vs {cat[filter_name][0]}")

    for idx, idv_flux in enumerate(cat[filter_name]):
        print(f"XXX obj {idx} flux = {idv_flux}")
        print(f"XXX obj {idx} mag (3631) = {-2.5 * np.log(idv_flux * 3631)}")
        print(f"XXX obj {idx} mag (abzero) = {-2.5 * (np.log(idv_flux))}")

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

    print(f"XXX moswcs = {moswcs}")

    # Simulate mosaic
    mosaic, extras = l3.simulate(context.shape[1:], moswcs, exptimes[0], filter_name, source_cat, metadata=metadata, seed=rng_seed)

    import plotly.express as px

    # fig1 = px.imshow(mosaic.data.value, title='1. Mosaic Data', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    # fig1.show()

    pos_vals = mosaic.data.value.copy()
    # pos_vals[pos_vals <= 0] = 1e-14
    pos_vals[pos_vals <= 0] = np.min(np.abs(pos_vals))

    # fig3 = px.imshow(np.log(pos_vals), title='3. Mosaic Data (log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    # fig3.show()


    # Second test, passing more in

    psf = romanisim.psf.make_psf(filter_name=filter_name, sca=CENTER_SCA, chromatic=False, webbpsf=True)

    efftimes = util.decode_context_times(context, exptimes)

    fig4 = px.imshow(efftimes, title='4. Effective Times', labels={'color': 'seconds', 'x': 'Y axis', 'y': 'X axis'})
    fig4.show()

    mosaic2, extras2 = l3.simulate(context.shape[1:], moswcs, efftimes, filter_name, source_cat, metadata=metadata, psf=psf, seed=rng_seed)

    # fig5 = px.imshow(mosaic2.data.value, title='5. Mosaic Data', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    # fig5.show()

    pos_vals2 = mosaic2.data.value.copy()
    # pos_vals2[pos_vals2 <= 0] = 0.00000000001
    pos_vals[pos_vals <= 0] = np.min(np.abs(pos_vals))

    # fig6 = px.imshow(np.log(pos_vals2), title='6. Mosaic Data (log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    # fig6.show()

   



    # The two tests below do not pass yet

    

    # true_locs = np.zeros((int(2*ydiff), int(2*xdiff)))
    true_locs = np.zeros((context.shape[1]+10, context.shape[2]+10))
    # Are there sources where there should be?
    # for r, d in zip(cat['ra'], cat['dec']):
        # x, y = moswcs.toImage(r, d, units=galsim.degrees)
    # print(f"XXX pos_vals2.shape = {pos_vals2.shape}")
    # print(f"XXX context.shape = {context.shape}")
    # print(f"XXX np.std(mosaic2.data.value) = {np.std(mosaic2.data.value)}")
    # print(f"XXX mean + std = {np.median(mosaic2.data.value) + np.std(mosaic2.data.value)}")
    # for x,y in zip(allx, ally):
    #     x = int(x - xmin)
    #     y = int(y - ymin)
    #     print(f"x, y = {x}, {y}")
    #     true_locs[y-10:y+10,x-10:x+10] = 1
    #     # assert mosaic2.data.value[y, x] > np.median(mosaic2.data.value) * 5
    # fig7 = px.imshow(true_locs, title='7. True locations', labels={'color': 'Found', 'x': 'Y axis', 'y': 'X axis'})
    # fig7.show()
    
    # print(f"coords = {coords}")
    # print(f"XXX moswcs = {moswcs}")
    # print(f"XXX twcs = {twcs}")
    # true_locs = np.zeros((int(ymax + 100), int(xmax + 100)))
    true_locs = np.zeros((context.shape[1]+10, context.shape[2]+10))
    # for r, d in coords:

    x_all, y_all = moswcs.radecToxy(coords[:, 0], coords[:, 1], 'rad')#units=galsim.degrees)
    for x,y in zip(x_all, y_all):
        x = int(x)
        y = int(y)
        print(f"x, y = {x}, {y}")
        true_locs[y-10:y+10,x-10:x+10] = 1
        # Undo this when flux scaling is fixed
        print(f"value / mean = {mosaic2.data.value[y, x]} / {np.median(mosaic2.data.value)} = {mosaic2.data.value[y, x] / np.median(mosaic2.data.value)}")
        assert mosaic2.data.value[y, x] > (np.median(mosaic2.data.value) * 5)
    fig8 = px.imshow(true_locs, title='8. True locations - by ra & dec', labels={'color': 'Found', 'x': 'Y axis', 'y': 'X axis'})
    fig8.show()

    fig9 = px.imshow(mosaic2.var_poisson.value, title='9. Poisson Variance ', labels={'color': 'MJy^2 / sr^2', 'x': 'Y axis', 'y': 'X axis'})
    fig9.show()

    fig10 = px.imshow(mosaic2.var_rnoise.value, title='10. Read Noise Variance ', labels={'color': 'MJy^2 / sr^2', 'x': 'Y axis', 'y': 'X axis'})
    fig10.show()

    fig11 = px.imshow(mosaic2.var_flat.value, title='11. Flat Noise Variance ', labels={'color': 'MJy^2 / sr^2', 'x': 'Y axis', 'y': 'X axis'})
    fig11.show()

    fig12 = px.imshow(mosaic2.err.value, title='12. Error ', labels={'color': 'MJy^2 / sr^2', 'x': 'Y axis', 'y': 'X axis'})
    fig12.show()

    # did we get all the flux?
    # Convert to CPS for comparison
    # Unit factor
    unit_factor = ((3631 * u.Jy) / (romanisim.bandpass.get_abflux(filter_name) * 10e6
                                    * parameters.reference_data['photom']["pixelareasr"][filter_name])).to(u.MJy / u.sr)
    totflux = np.sum(mosaic2.data.value - np.median(mosaic2.data.value)) / unit_factor.value

    # Flux to counts
    cps_conv = romanisim.bandpass.get_abflux(filter_name)
    # expectedflux = (romanisim.bandpass.get_abflux(filter_name) * np.sum(cat[filter_name])
    #                 / parameters.reference_data['gain'].value)
    # expectedflux = np.sum(cat[filter_name]) / parameters.reference_data['gain'].value
    # expectedflux = ((np.sum(cat[filter_name]) * 1e6)
    #                 / (parameters.reference_data['gain'].value * 3631))
    # expectedflux = ((np.sum(cat[filter_name]) * 3631)
    #                 / (parameters.reference_data['gain'].value * 1e6))
    expectedflux = np.sum(cat[filter_name]) * cps_conv
    print(f"XXX totflux = {totflux}")
    print(f"XXX expectedflux = {expectedflux}")
    print(f"XXX totflux / expectedflux = {totflux / expectedflux}")
    print(f"XXX expectedflux / totflux= {expectedflux / totflux}")
    print(f"XXX romanisim.bandpass.get_abflux(filter_name) = {romanisim.bandpass.get_abflux(filter_name)}")
    print(f"XXX np.sum(cat[filter_name]) = {np.sum(cat[filter_name])}")
    print(f"XXX np.sum(cat[filter_name])/3631 = {np.sum(cat[filter_name])/3631}")
    # Something about the flux scaling or the sky is off? This fails
    assert np.abs(np.log(expectedflux) / np.log(totflux) - 1) < 0.1


def test_simulate_vs_cps():
    rng_seed = 42
    imdict = set_up_image_rendering_things()
    chromcat = imdict['chromcatalog']
    graycat = imdict['graycatalog']
    coord = SkyCoord(270 * u.deg, 66 * u.deg)
    for o in chromcat:
        o.sky_pos = coord
    for o in graycat:
        o.sky_pos = coord
    # these are all dumb coordinates; the coord sent to simulate_counts
    # is the coordinate of the boresight, but that doesn't need to be on SCA 1.
    # But at least they'll exercise some machinery if the ignore_distant_sources
    # argument is high enough!
    roman.n_pix = 100
    # efftimes = 600

    meta = util.default_image_meta(filter_name='F158')
    wcs.fill_in_parameters(meta, coord)

    # imdict = set_up_image_rendering_things()
    im = imdict['im'].copy()
    im.array[:] = 0
    # npix = np.prod(im.array.shape)
    exptime = 600
    zpflux = 10

    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['basic']['optical_element'] = filter_name
    metadata['wcsinfo']['ra_ref'] = 270
    metadata['wcsinfo']['dec_ref'] = 66
    # coord = SkyCoord(270 * u.deg, 66 * u.deg)
    # wcs.fill_in_parameters(metadata, coord)
    twcs = romanisim.wcs.get_mosaic_wcs(metadata, shape=(roman.n_pix, roman.n_pix))

    im1 = im.copy()
    # _, objinfo1 = l3.simulate_cps(im1, metadata, exptime, objlist=imdict['chromcatalog'], ignore_distant_sources=100)
    im1, extras1 = l3.simulate_cps(im1, metadata, exptime, objlist=chromcat,
                                  xpos=[50] * len(chromcat), ypos=[50] * len(chromcat),
                                    # zpflux=zpflux, 
                                    # psf=imdict['impsfchromatic'],  
                                  bandpass=imdict['bandpass'],
                                #   coords_unit='deg',
                                 seed=rng_seed,
                                  ignore_distant_sources=100)

    im2 = im.copy()
    im2, extras2 = l3.simulate_cps(im2, metadata, exptime, objlist=graycat, 
                                   xpos=[50] * len(chromcat), ypos=[50] * len(chromcat),
                                #    coords_unit='deg',
                                    psf=imdict['impsfgray'],
                                seed=rng_seed, 
                                   ignore_distant_sources=100)

    import plotly.express as px

    # fig1 = px.imshow(im1.array, title='1. im1.array', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    # fig1.show()

    # pos_vals = im1.array.copy()
    # pos_vals[pos_vals <= 0] = np.min(np.abs(pos_vals))

    # fig2 = px.imshow(np.log(pos_vals), title='2. im1.array(log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    # fig2.show()

    # fig3 = px.imshow(im2.array, title='3. im2.array', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    # fig3.show()

    # pos_vals = im2.array.copy()
    # pos_vals[pos_vals <= 0] = np.min(np.abs(pos_vals))

    # fig4 = px.imshow(np.log(pos_vals), title='4. im2.array(log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    # fig4.show()

    # im1 = im1[0].array
    # im2 = im2[0].array
    # maxim = np.where(im1 > im2, im1, im2)
    # m = np.abs(im1 - im2) <= 20 * np.sqrt(maxim)
    maxim = np.where(im1.array**2 > im2.array**2, im1.array, im2.array)
    print(f"XXXX np.max(np.abs(im1.array - im2.array)) = {np.max(np.abs(im1.array - im2.array))}")
    print(f"XXXX maxim = {maxim}")
    # m = np.abs(im1.array - im2.array) <= 20 * np.sqrt(maxim)
    m = np.abs(im1.array**2 - im2.array**2) <= 20**2 * np.abs(maxim)

    fig10 = px.imshow(np.abs(im1.array - im2.array), title='10. Diff ', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    fig10.show()

    assert np.all(m)

    # im1 = im.copy()
    # _, objinfo1 = l3.simulate(im1, metadata, exptime, objlist=chromcat,
    #                               xpos=[50] * len(chromcat), ypos=[50] * len(chromcat),
    #                               zpflux=zpflux, psf=imdict['impsfchromatic'], bandpass=imdict['bandpass'],
    #                               ignore_distant_sources=100)
    
    # Simulate mosaic
    # im1, extras1 = l3.simulate(context.shape[1:], moswcs, exptimes[0], filter_name, source_cat, metadata=metadata)

    print(f"XXXX twcs = {twcs}")
    print(f"XXX roman.n_pix = {roman.n_pix}")

    im3, extras1 = l3.simulate((roman.n_pix, roman.n_pix), twcs, exptime, filter_name, chromcat,
                                # psf=imdict['impsfchromatic'],
                               bandpass=imdict['bandpass'], 
                                # zpflux=zpflux,
                                seed=rng_seed,
                                cps_conv=1, unit_factor=(1 * u.MJy / u.sr),
                               metadata=metadata, sky=0,
                               ignore_distant_sources=100,
                               effreadnoise=0,
                               )
    im4, extras1 = l3.simulate((roman.n_pix, roman.n_pix), twcs, exptime, filter_name, graycat,
                               psf=imdict['impsfgray'],
                               cps_conv=1, unit_factor=(1 * u.MJy / u.sr),
                            seed=rng_seed,
                               metadata=metadata, sky=0,
                               ignore_distant_sources=100,
                               effreadnoise=0,
                               )


    # fig5 = px.imshow(im1['data'].value, title='5. im1.[data].value', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    # fig5.show()

    # pos_vals = im1['data'].value.copy()
    # pos_vals[pos_vals <= 0] = np.min(np.abs(pos_vals))

    # fig6 = px.imshow(np.log(pos_vals), title='6. im1.[data].value (log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    # fig6.show()

    # fig7 = px.imshow(im2['data'].value, title='7. im2.[data].value', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    # fig7.show()

    # pos_vals = im2['data'].value.copy()
    # pos_vals[pos_vals <= 0] = np.min(np.abs(pos_vals))

    # fig8 = px.imshow(np.log(pos_vals), title='8. im2.[data].value (log)', labels={'color': 'log(MJy / sr)', 'x': 'Y axis', 'y': 'X axis'})
    # fig8.show()

    # maxim = np.where(im3['data'].value > im4['data'].value, im3['data'].value, im4['data'].value)
    # m = np.abs(im3['data'].value - im4['data'].value) <= 20 * np.sqrt(maxim)

    maxim = np.where(im3['data'].value**2 > im4['data'].value**2, im3['data'].value, im4['data'].value)
    m = np.abs(im3['data'].value**2 - im4['data'].value**2) <= 20**2 * np.abs(maxim)

    print(f"XXXX np.max(np.abs(im3['data'].value - im4['data'].value)) = {np.max(np.abs(im3['data'].value - im4['data'].value))}")
    print(f"XXXX maxim = {maxim}")

    fig11 = px.imshow(np.abs(im1.array - im2.array), title='11. Diff ', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    fig11.show()

    fig12 = px.imshow(m, title='12. Diff boolean', labels={'color': 'MJy / sr', 'x': 'Y axis', 'y': 'X axis'})
    fig12.show()

    assert np.all(m)

    assert np.allclose(im1.array, im3['data'].value)
    assert np.allclose(im2.array, im4['data'].value)


def test_simulate_cps():
    rng_seed = 42
    imdict = set_up_image_rendering_things()
    im = imdict['im'].copy()
    im.array[:] = 0
    npix = np.prod(im.array.shape)
    exptime = 100
    zpflux = 10   
    
    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['basic']['optical_element'] = filter_name
    metadata['wcsinfo']['ra_ref'] = 270
    metadata['wcsinfo']['dec_ref'] = 66
    coord = SkyCoord(270 * u.deg, 66 * u.deg)
    wcs.fill_in_parameters(metadata, coord)

    # Test empty image
    l3.simulate_cps(
        im, metadata, exptime, objlist=[], psf=imdict['impsfgray'],
        sky=0, zpflux=zpflux)
    assert np.all(im.array == 0)  # verify nothing in -> nothing out

    # Test flat sky
    sky = im.copy()
    skycountspersecond = 1
    sky.array[:] = skycountspersecond
    im2 = im.copy()
    l3.simulate_cps(im2, metadata, exptime, sky=sky, seed=rng_seed, zpflux=zpflux)
    # verify adding the sky increases the counts
    assert np.all(im2.array >= im.array)
    # verify that the count rate is about right.
    # poisson_rate = skycountspersecond * exptime
    poisson_rate = skycountspersecond
    print(f"XXX np.mean(im2.array) = {np.mean(im2.array)}")
    assert (np.abs(np.mean(im2.array) - poisson_rate)
            < 10 * np.sqrt(poisson_rate / npix))
     
    # verify that Poisson noise is included
    # pearson chi2 test is probably best here, but it's finicky to get
    # right---one needs to choose the right bins so that the convergence
    # to the chi^2 distribution is close enough.
    # For a Poisson distribution, the variance is equal to the mean rate;
    # let's verify that in fact the variance matches expectations within
    # some tolerance.
    # the variance on the sample variance for a Gaussian is 2*sigma^4/(N-1)
    # this isn't a Gaussian but should be close with 100 counts?
    var_of_var = 2 * (poisson_rate ** 2) * exptime / (npix - 1)
    print(f"XXX poisson_rate = {poisson_rate}")
    print(f"XXX var_of_var = {var_of_var}")
    print(f"XXX np.var(im2.array) = {np.var(im2.array)}")
    assert (np.abs(poisson_rate - np.var(im2.array))
            < 10 * np.sqrt(var_of_var))
    log.info('DMS230: successfully included Poisson noise in image.')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'image': im2.array,
                   'poisson_rate': poisson_rate,
                   'variance_of_variance': var_of_var,
                   }
        af.write_to(os.path.join(artifactdir, 'dms230.asdf'))

    # No darks involved.. these should be removed
    # im3 = im.copy()
    # l3.simulate_cps(im3, metadata, exptime, dark=sky, zpflux=zpflux)
    # # verify that the dark counts don't see the zero point conversion
    # assert (np.abs(np.mean(im3.array) - exptime)
    #         < 20 * np.sqrt(skycountspersecond * exptime / npix))
    # im4 = im.copy()
    # l3.simulate_cps(im4, metadata, exptime, dark=sky, flat=0.5,
    #                               zpflux=zpflux)
    # # verify that dark electrons are not hit by the flat
    # assert np.all(im3.array - im4.array
    #               < 20 * np.sqrt(exptime * skycountspersecond))

    # No flats involved.. these should be removed
    # im5 = im.copy()
    # l3.simulate_cps(im5, metadata, exptime, sky=sky, flat=0.5,
    #                               zpflux=zpflux)
    # # verify that sky photons are hit by the flat
    # poisson_rate = skycountspersecond * exptime * 0.5
    # assert (np.abs(np.mean(im5.array) - poisson_rate)
    #         < 20 * np.sqrt(poisson_rate / npix))
    

    # there are a few WCS bits where we use the positions in the catalog
    # to figure out where to render objects.  That would require setting
    # up a real PSF and is annoying.  Skipping that.
    # render some objects
    print("XXX im3 block")
    im3 = im.copy()
    _, objinfo = l3.simulate_cps(
        im3, metadata, exptime, objlist=imdict['graycatalog'], psf=imdict['impsfgray'],
        xpos=[50, 50], ypos=[50, 50], seed=rng_seed, zpflux=zpflux)
        # filter_name=imdict['filter_name'])
    assert np.sum(im3.array) > 0  # at least verify that we added some sources...
    print(f"XXX objinfo = {objinfo['objinfo']}")
    assert len(objinfo['objinfo']['array']) == 2  # two sources were added
    print("XXX im4 block")
    im4 = im.copy()
    _, objinfo = l3.simulate_cps(
        im4, metadata, exptime, objlist=imdict['chromcatalog'],
        xpos=[50, 50], ypos=[50, 50],
        seed=rng_seed,
        psf=imdict['impsfchromatic'], bandpass=imdict['bandpass'])
    assert np.sum(im4.array) > 0  # at least verify that we added some sources...
    assert len(objinfo['objinfo']['array']) == 2  # two sources were added
    print("XXX im5 block")
    im5 = im.copy()
    _, objinfo = l3.simulate_cps(
        im5, metadata, exptime, objlist=imdict['chromcatalog'],
        psf=imdict['impsfchromatic'], xpos=[1000, 1000],
        seed=rng_seed,
        ypos=[1000, 1000], zpflux=zpflux)
    # print(f"XXX objinfo = {objinfo}")
    # print(f"XXX objinfo['objinfo'].dtype.metadata = {objinfo['objinfo'].dtype['counts']}")
    # for key in objinfo.dtype.metadata.keys():
    #     print(f"XXX key = {key}")
    assert np.sum(objinfo['objinfo']['counts'] > 0) == 0
    # these sources should be out of bounds


def test_make_l3():
    # resultants = np.ones((4, 50, 50), dtype='i4')
    # read_pattern = [[1 + x for x in range(4)],
    #                 [5 + x for x in range(4)],
    #                 [9 + x for x in range(4)],
    #                 [13 + x for x in range(4)]]
    # slopes, readvar, poissonvar = image.make_l2(
    #     resultants, read_pattern, gain=1, flat=1, darkrate=0)
    
    rng_seed = 42
    imdict = set_up_image_rendering_things()
    im = imdict['im'].copy()
    im.array[:] = 0
    npix = np.prod(im.array.shape)
    exptime = 100
    zpflux = 10   
    
    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)

    mosaic_mdl1 = l3.make_l3(im, metadata, exptime)

    assert np.allclose(mosaic_mdl1['data'].value, 0)


    # resultants[:, :, :] = np.arange(4)[:, None, None]
    # resultants *= u.DN
    # gain = 1 * u.electron / u.DN
    # slopes, readvar, poissonvar = image.make_l2(
    #     resultants, read_pattern,
    #     gain=gain, flat=1, darkrate=0)
    
    im.array[:] = 4
    mosaic_mdl2 = l3.make_l3(im, metadata, exptime)

    assert np.allclose(mosaic_mdl2['data'], 4 * u.MJy / u.sr)
    assert np.all(np.array(mosaic_mdl2['data'].shape) == np.array(mosaic_mdl2['err'].shape))
    assert np.all(np.array(mosaic_mdl2['data'].shape) == np.array(mosaic_mdl2['context'][0,:].shape))
    assert np.all(np.array(mosaic_mdl2['data'].shape) == np.array(mosaic_mdl2['weight'].shape))
    assert np.all(np.array(mosaic_mdl2['data'].shape) == np.array(mosaic_mdl2['var_poisson'].shape))
    assert np.all(np.array(mosaic_mdl2['data'].shape) == np.array(mosaic_mdl2['var_rnoise'].shape))
    assert np.all(np.array(mosaic_mdl2['data'].shape) == np.array(mosaic_mdl2['var_flat'].shape))
    assert np.all(mosaic_mdl2['var_poisson'].value >= 0)
    assert np.all(mosaic_mdl2['weight'] >= 0)
    assert np.allclose(mosaic_mdl2['err'].value**2, 
                       mosaic_mdl2['var_poisson'].value + mosaic_mdl2['var_rnoise'].value + mosaic_mdl2['var_flat'].value)

    # slopes1, readvar1, poissonvar1 = image.make_l2(
    #     resultants, read_pattern, read_noise=1, darkrate=0)
    # slopes2, readvar2, poissonvar2 = image.make_l2(
    #     resultants, read_pattern, read_noise=2, darkrate=0)
    # assert np.all(readvar2 >= readvar1)
    # # because we change the weights depending on the ratio of read & poisson
    # # noise, we can't assume above that readvar2 = readvar1 * 4.
    # # But it should be pretty darn close here.
    # assert np.all(np.abs(readvar2 / (readvar1 * 4) - 1) < 0.1)
    # slopes2, readvar2, poissonvar2 = image.make_l2(
    #     resultants, read_pattern, read_noise=1, flat=0.5, darkrate=0)
    # assert np.allclose(slopes2, slopes1 / 0.5)
    # assert np.allclose(readvar2, readvar1 / 0.5**2)
    # assert np.allclose(poissonvar2, poissonvar1 / 0.5**2)
    # slopes, readvar, poissonvar = image.make_l2(
    #     resultants, read_pattern, read_noise=1, gain=gain, flat=1,
    #     darkrate=1 / parameters.read_time / 4)
    # assert np.allclose(slopes, 0, atol=1e-6)
    
    # Expand this to include simulation? More complext read / context?


def test_exptime_array():
    rng_seed = 42
    imdict = set_up_image_rendering_things()
    chromcat = [imdict['chromcatalog'][0]]
    graycat = [imdict['graycatalog'][0]]
    coord = SkyCoord(270 * u.deg, 66 * u.deg)
    for o in chromcat:
        o.sky_pos = coord
    for o in graycat:
        o.sky_pos = coord
    # these are all dumb coordinates; the coord sent to simulate_counts
    # is the coordinate of the boresight, but that doesn't need to be on SCA 1.
    # But at least they'll exercise some machinery if the ignore_distant_sources
    # argument is high enough!
    roman.n_pix = 100

    meta = util.default_image_meta(filter_name='F158')
    wcs.fill_in_parameters(meta, coord)

    # imdict = set_up_image_rendering_things()
    im = imdict['im'].copy()
    im.array[:] = 0
    # npix = np.prod(im.array.shape)
    exptime = np.ones((roman.n_pix, roman.n_pix))
    exptime[0:50, :] = 300
    exptime[50:, :] = 600

    # zpflux = 10

    metadata = copy.deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['basic']['optical_element'] = filter_name
    metadata['wcsinfo']['ra_ref'] = 270
    metadata['wcsinfo']['dec_ref'] = 66
    # coord = SkyCoord(270 * u.deg, 66 * u.deg)
    # wcs.fill_in_parameters(metadata, coord)
    twcs = romanisim.wcs.get_mosaic_wcs(metadata, shape=(roman.n_pix, roman.n_pix))

    import plotly.express as px

    # im1 = im.copy()
    # _, objinfo1 = l3.simulate(im1, metadata, exptime, objlist=chromcat,
    #                               xpos=[50] * len(chromcat), ypos=[50] * len(chromcat),
    #                               zpflux=zpflux, psf=imdict['impsfchromatic'], bandpass=imdict['bandpass'],
    #                               ignore_distant_sources=100)
    
    # Simulate mosaic
    # im1, extras1 = l3.simulate(context.shape[1:], moswcs, exptimes[0], filter_name, source_cat, metadata=metadata)

    print(f"XXXX twcs = {twcs}")
    print(f"XXX roman.n_pix = {roman.n_pix}")

    im1, extras1 = l3.simulate((roman.n_pix, roman.n_pix), twcs, exptime, filter_name, chromcat,
                                # psf=imdict['impsfchromatic'],
                               bandpass=imdict['bandpass'], 
                                # zpflux=zpflux,
                                seed=rng_seed,
                                cps_conv=1, unit_factor=(1 * u.MJy / u.sr),
                               metadata=metadata,
                            #    sky=0,
                               ignore_distant_sources=100,
                               )
    im2, extras2 = l3.simulate((roman.n_pix, roman.n_pix), twcs, exptime, filter_name, graycat,
                               psf=imdict['impsfgray'],
                               cps_conv=1, unit_factor=(1 * u.MJy / u.sr),
                            seed=rng_seed,
                               metadata=metadata,
                            #    sky=0,
                               ignore_distant_sources=100,
                               )


    fig5 = px.imshow(im2['var_poisson'].value, title='5. im2.[var_poisson].value', labels={'color': 'MJy^2 / sr^2', 'x': 'Y axis', 'y': 'X axis'})
    fig5.show()

    # print(f"XXX np.mean(im1[var_poisson][0:50, :].value) = {np.mean(im1['var_poisson'][0:50, :].value)}")
    # print(f"XXX np.mean(im1[var_poisson][50:, :].value) = {np.mean(im1['var_poisson'][50:, :].value)}")
    # print(f"XXX vp ratio = {(np.mean(im1['var_poisson'][0:50, :].value) / np.mean(im1['var_poisson'][50:, :].value))}")

    assert np.isclose((np.mean(im1['var_poisson'][0:50, :].value) / np.mean(im1['var_poisson'][50:, :].value)), 2**0.5, rtol=0.1)

    # print(f"XXX np.mean(im2[var_poisson][0:50, :].value) = {np.mean(im2['var_poisson'][0:50, :].value)}")
    # print(f"XXX np.mean(im2[var_poisson][50:, :].value) = {np.mean(im2['var_poisson'][50:, :].value)}")
    # print(f"XXX vp ratio = {(np.mean(im2['var_poisson'][0:50, :].value) / np.mean(im2['var_poisson'][50:, :].value))}")

    assert np.isclose((np.mean(im2['var_poisson'][0:50, :].value) / np.mean(im2['var_poisson'][50:, :].value)), 2**0.5, rtol=0.1)

    # print(f"XXX np.mean(im1[data][0:50, :].value) = {np.mean(im1['data'][0:50, :].value)}")
    # print(f"XXX np.mean(im1[data]][50:, :].value) = {np.mean(im1['data'][50:, :].value)}")

    # print(f"XXX adjusted np.mean(im1[data][0:50, :].value) = {np.mean(im1['data'][0:50, :].value - im1['var_poisson'][0:50, :].value)}")
    # print(f"XXX adjusted np.mean(im1[data]][50:, :].value) = {np.mean(im1['data'][50:, :].value - im1['var_poisson'][50:, :].value)}")

    # print(f"XXX np.mean(im2[data][0:50, :].value) = {np.mean(im2['data'][0:50, :].value)}")
    # print(f"XXX np.mean(im2[data]][50:, :].value) = {np.mean(im2['data'][50:, :].value)}")

    # print(f"XXX adjusted np.mean(im2[data][0:50, :].value) = {np.mean(im2['data'][0:50, :].value - im2['var_poisson'][0:50, :].value)}")
    # print(f"XXX adjusted np.mean(im2[data]][50:, :].value) = {np.mean(im2['data'][50:, :].value - im2['var_poisson'][50:, :].value)}")

    assert np.isclose(np.mean(im1['data'][0:50, :].value), np.mean(im1['data'][50:, :].value), rtol=0.2)
    assert np.isclose(np.mean(im2['data'][0:50, :].value), np.mean(im2['data'][50:, :].value), rtol=0.2)

# TBD: Test of geometry construction
