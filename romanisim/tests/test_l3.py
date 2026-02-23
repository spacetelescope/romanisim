"""Unit tests for mosaic module.

"""

import os
from copy import deepcopy
import math
import numpy as np
import galsim
from romanisim import catalog, l3, util, log
from romanisim.models import parameters, wcs, psf
from astropy import units as u
from astropy import table
from astropy.stats import mad_std
import asdf
import pytest
from roman_datamodels import datamodels as rdm
import romanisim.models.bandpass
from astropy.coordinates import SkyCoord


@pytest.mark.soctests
def test_inject_sources_into_mosaic():
    """Inject sources into a mosaic.
    """

    # Set constants and metadata
    parameters.n_pix = 200
    rng_seed = 42
    metadata = deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['instrument']['optical_element'] = filter_name

    # Create WCS
    twcs = wcs.GWCS(wcs.get_mosaic_wcs(
        metadata, shape=(parameters.n_pix, parameters.n_pix)))

    # Create initial Level 3 mosaic

    # Create Four-quadrant pattern of gaussian noise, centered around one
    # Each quadrant's gaussian noise scales like total exposure time
    # (total files contributed to each quadrant)

    # Create gaussian noise generators
    # sky should generate ~0.2 electron / s / pix.
    # MJy / sr has similar magnitude to electron / s (i.e., within a factor
    # of several), so just use 0.2 here.
    meanflux = 0.2
    g1 = galsim.GaussianDeviate(rng_seed, mean=meanflux, sigma=0.01 * meanflux)
    g2 = galsim.GaussianDeviate(rng_seed, mean=meanflux, sigma=0.02 * meanflux)
    g3 = galsim.GaussianDeviate(rng_seed, mean=meanflux, sigma=0.05 * meanflux)
    g4 = galsim.GaussianDeviate(rng_seed, mean=meanflux, sigma=0.10 * meanflux)

    # Create level 3 mosaic model
    l3_mos = rdm.MosaicModel.create_fake_data(shape=(parameters.n_pix, parameters.n_pix))
    l3_mos['meta']['wcs'] = twcs._wcs

    # Update metadata in the l3 model
    for key in metadata.keys():
        if key in l3_mos.meta:
            l3_mos.meta[key].update(metadata[key])

    # Obtain unit conversion factors
    # maggies to counts (large number)
    sca = parameters.default_sca
    cps_conv = romanisim.models.bandpass.get_abflux(filter_name, sca)
    # electrons to mjysr (roughly order unity in scale)
    unit_factor = romanisim.models.bandpass.etomjysr(filter_name, sca)

    # Populate the mosaic data array with gaussian noise from generators
    g1.generate(l3_mos.data[0:100, 0:100])
    g2.generate(l3_mos.data[0:100, 100:200])
    g3.generate(l3_mos.data[100:200, 0:100])
    g4.generate(l3_mos.data[100:200, 100:200])

    # Define Poisson Noise of mosaic
    l3_mos.var_poisson[0:100, 0:100] = 0.01**2 * meanflux**2
    l3_mos.var_poisson[0:100, 100:200] = 0.02**2 * meanflux**2
    l3_mos.var_poisson[100:200, 0:100] = 0.05**2 * meanflux**2
    l3_mos.var_poisson[100:200, 100:200] = 0.1**2 * meanflux**2

    # Create normalized psf source catalog (same source in each quadrant)
    mag_flux = 1e-9
    sc_dict = {"ra": 4 * [0.0], "dec": 4 * [0.0], "type": 4 * ["PSF"], "n": 4 * [-1.0],
               "half_light_radius": 4 * [0.0], "pa": 4 * [0.0], "ba": 4 * [1.0], filter_name: 4 * [mag_flux]}
    sc_table = table.Table(sc_dict)

    # Set locations
    xpos_idx = [50, 50, 150, 150]
    ypos_idx = [50, 150, 50, 150]

    ra, dec = twcs._radec(np.array(xpos_idx), np.array(ypos_idx))
    sc_table['ra'] = np.degrees(ra)
    sc_table['dec'] = np.degrees(dec)

    # Copy original Mosaic before adding sources as sources are added in place
    l3_mos_orig = l3_mos.copy()
    l3_mos_orig.data = l3_mos.data.copy()
    l3_mos_orig.var_poisson = l3_mos.var_poisson.copy()

    # Add source_cat objects to mosaic
    l3_mos = l3.inject_sources_into_l3(
        l3_mos, sc_table, seed=rng_seed)

    # Ensure that every data pixel value has increased or
    # remained the same with the new sources injected
    assert np.all(l3_mos.data >= l3_mos_orig.data)

    # Ensure that every pixel's poisson variance has increased or
    # remained the same with the new sources injected
    # Numpy isclose is needed to determine equality, due to float precision issues
    close_mask = np.isclose(l3_mos.var_poisson, l3_mos_orig.var_poisson, rtol=1e-06)
    assert False in close_mask
    assert np.all(l3_mos.var_poisson[~close_mask] > l3_mos_orig.var_poisson[~close_mask])

    # Ensure total added flux matches expected added flux
    total_rec_flux = np.sum(l3_mos.data - l3_mos_orig.data)  # MJy / sr
    total_theo_flux = 4 * mag_flux * cps_conv * unit_factor  # u.MJy / u.sr
    assert np.isclose(total_rec_flux, total_theo_flux, rtol=4e-02)

    # Create log entry and artifacts
    log.info('DMS232 successfully injected sources into a mosaic at points (50,50), (50,150), (150,50), (150,150).')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'l3_mos': l3_mos._instance,
                   'l3_mos_orig': l3_mos_orig._instance,
                   'source_cat_table': sc_table,
                   }
        af.write_to(os.path.join(artifactdir, 'dms232.asdf'))


@pytest.mark.soctests
def test_sim_mosaic():
    """Generating mosaic from catalog file.
    """
    # Define random seed
    rng_seed = 42

    # Obtain pointing
    ra_ref = parameters.default_mosaic_parameters_dictionary['wcsinfo']['ra_ref']
    dec_ref = parameters.default_mosaic_parameters_dictionary['wcsinfo']['dec_ref']

    # Set metadata and capture filter
    metadata = deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = metadata['instrument']['optical_element']

    # Setting the SCA for proper flux calculations
    sca = parameters.default_sca

    # Set exposure time
    exptimes = [600]

    # Create catalog of objects
    cen = SkyCoord(ra=(ra_ref * u.deg).to(u.rad), dec=(dec_ref * u.deg).to(u.rad))
    cat = catalog.make_dummy_table_catalog(cen, radius=0.02, nobj=100, seed=rng_seed)
    # Make the first 10 bright for tests
    cat[filter_name][0:10] *= 1e4

    # Create bounds from the object list
    twcs = romanisim.models.wcs.get_mosaic_wcs(metadata)
    allx, ally = twcs.world_to_pixel_values(cat['ra'].value, cat['dec'].value)

    # Obtain the sample extremums
    xmin = min(allx)
    xmax = max(allx)
    ymin = min(ally)
    ymax = max(ally)

    # Obtain WCS center
    xcen, ycen = twcs.world_to_pixel_values(ra_ref, dec_ref)

    # Determine maximum extremums from WCS center
    xdiff = max([math.ceil(xmax - xcen), math.ceil(xcen - xmin)]) + 1
    ydiff = max([math.ceil(ymax - ycen), math.ceil(ycen - ymin)]) + 1

    # Create context map preserving WCS center
    context = np.ones((1, 2 * ydiff, 2 * xdiff), dtype=np.uint32)

    # Generate properly sized WCS
    moswcs = romanisim.models.wcs.get_mosaic_wcs(metadata, shape=(context.shape[1:]))

    # Simulate mosaic
    mosaic, extras = l3.simulate(context.shape[1:], moswcs, exptimes[0],
                                 filter_name, cat, metadata=metadata,
                                 seed=rng_seed)

    # Did all sources get simulated?
    assert len(extras['simcatobj']) == len(cat)

    # Ensure center pixel of bright objects is bright
    x_all, y_all = moswcs.world_to_pixel_values(cat['ra'][:10].value,
                                                cat['dec'][:10].value)
    for x, y in zip(x_all, y_all):
        x = int(x)
        y = int(y)
        assert mosaic.data[y, x] > (np.median(mosaic.data) * 5)

    # Did we get all the flux?
    etomjysr = romanisim.models.bandpass.etomjysr(filter_name, sca)
    totflux = np.sum(mosaic.data - np.median(mosaic.data)) / etomjysr

    # Flux to counts
    cps_conv = romanisim.models.bandpass.get_abflux(filter_name, sca)
    expectedflux = np.sum(cat[filter_name]) * cps_conv

    # Ensure that the measured flux is close to the expected flux
    assert np.abs(np.log(expectedflux) / np.log(totflux) - 1) < 0.1

    # Is the noise about right?
    assert np.abs(
        mad_std(mosaic.data) / np.median(mosaic.err) - 1) < 0.5
    # note large ~50% error bar there; value in initial test run is 0.25
    # a substantial number of source pixels have flux, so the simple medians
    # and mads aren't terribly right.
    # if I repeat this after only including the first source I get 1.004.

    af = asdf.AsdfFile()
    af.tree = {'roman': mosaic}
    af.validate()

    # Add log entries and artifacts
    log.info('DMS219 successfully created mosaic file with sources rendered '
             'at correct locations with matching flux and added noise.')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'l3_mosaic': mosaic,
                   'source_cat_table': cat,
                   }
        af.write_to(os.path.join(artifactdir, 'dms219.asdf'))


def set_up_image_rendering_things():
    """ Function to set up objects for use in tests
    """
    # Create sample image, filter, etc.
    im = galsim.ImageF(100, 100, scale=0.11, xmin=0, ymin=0)
    filter_name = 'F158'
    sca = 1
    impsfgray = psf.make_psf(sca, filter_name, psftype='epsf', chromatic=False)
    impsfchromatic = psf.make_psf(sca, filter_name, psftype='galsim',
                                  chromatic=True)
    bandpass = romanisim.models.bandpass.getBandpasses(AB_zeropoint=True)['H158']
    counts = 1000
    maggiestoe = romanisim.models.bandpass.get_abflux(filter_name, sca)
    fluxdict = {filter_name: counts}
    fluxdictgray = {filter_name: counts / maggiestoe}

    # Sample catalogs
    graycatalog = [
        catalog.CatalogObject(None, galsim.DeltaFunction(),
                              deepcopy(fluxdictgray)),
        catalog.CatalogObject(None, galsim.Sersic(1, half_light_radius=0.2),
                              deepcopy(fluxdictgray))
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
    tabcat[filter_name] = counts / maggiestoe
    tabcat['type'] = 'PSF'
    tabcat['n'] = -1
    tabcat['half_light_radius'] = -1
    tabcat['pa'] = -1
    tabcat['ba'] = -1

    # Return dictionary with the above values
    return dict(im=im, impsfgray=impsfgray,
                impsfchromatic=impsfchromatic,
                bandpass=bandpass, counts=counts, fluxdict=fluxdict,
                graycatalog=graycatalog,
                chromcatalog=chromcatalog, filter_name=filter_name, sca=sca,
                tabcatalog=tabcat)


def test_simulate_vs_cps():
    """ Tests to ensure that simulate runs match simulate_cps output
    """
    # Set random seed
    rng_seed = 42

    # Set image, catalog, etc.
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
    parameters.n_pix = 100
    exptime = 600

    # Create metadata
    meta = util.default_image_meta(filter_name='F158')
    wcs.fill_in_parameters(meta, coord)
    metadata = deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['instrument']['optical_element'] = filter_name
    metadata['wcsinfo']['ra_ref'] = 270
    metadata['wcsinfo']['dec_ref'] = 66
    # Adding the detector information as the simulations now support all 18 detectors with their own throughput curves
    # Using the default detector from the default_parameters_dictionary as all sca arguments are set to it within l3.py
    sca = parameters.default_sca

    # Set up blank image
    im = imdict['im'].copy()
    im.array[:] = 0

    maggytoes = romanisim.models.bandpass.get_abflux(filter_name, sca)
    etomjysr = romanisim.models.bandpass.etomjysr(filter_name, sca)

    twcs = wcs.get_mosaic_wcs(meta, shape=im.array.shape)
    im.wcs = wcs.GWCS(twcs)

    # Create chromatic data in simulate_cps

    im1 = im.copy()
    im1, extras1 = l3.simulate_cps(im1, filter_name, exptime, objlist=chromcat,
                                   bandpass=imdict['bandpass'],
                                   psf=imdict['impsfchromatic'],
                                   seed=rng_seed,
                                   ignore_distant_sources=100,
                                   maggytoes=maggytoes, etomjysr=etomjysr)

    # Create filter data in simulate_cps
    im2 = im.copy()
    im2, extras2 = l3.simulate_cps(im2, filter_name, exptime, objlist=graycat,
                                   psf=imdict['impsfgray'], seed=rng_seed,
                                   ignore_distant_sources=100,
                                   maggytoes=maggytoes, etomjysr=etomjysr)

    # Create chromatic data in simulate
    im3, extras3 = l3.simulate((parameters.n_pix, parameters.n_pix), twcs, exptime, filter_name, chromcat,
                               bandpass=imdict['bandpass'],
                               psf=imdict['impsfchromatic'],
                               seed=rng_seed,
                               metadata=metadata, sky=0,
                               ignore_distant_sources=100,
                               effreadnoise=0,
                               )

    # Create filter data in simulate
    im4, extras4 = l3.simulate((parameters.n_pix, parameters.n_pix), twcs, exptime, filter_name, graycat,
                               psf=imdict['impsfgray'],
                               seed=rng_seed,
                               metadata=metadata, sky=0,
                               ignore_distant_sources=100,
                               effreadnoise=0,
                               )

    # Ensure that the simulate and simulate_cps output matches for each type
    assert np.allclose(im1.array, im3['data'])
    assert np.allclose(im2.array, im4['data'])


def test_simulate_cps():
    """ Test various simulation options
    """
    # Set random seed
    rng_seed = 42

    # Set image, catalog, etc.
    imdict = set_up_image_rendering_things()
    im = imdict['im'].copy()
    im.array[:] = 0
    npix = np.prod(im.array.shape)
    exptime = 100

    # Create metadata
    metadata = deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['instrument']['optical_element'] = filter_name
    metadata['wcsinfo']['ra_ref'] = 270
    metadata['wcsinfo']['dec_ref'] = 66
    coord = SkyCoord(270 * u.deg, 66 * u.deg)
    wcs.fill_in_parameters(metadata, coord)

    # Test empty image
    l3.simulate_cps(
        im, filter_name, exptime, objlist=[], psf=imdict['impsfgray'],
        sky=0)
    assert np.all(im.array == 0)  # verify nothing in -> nothing out

    # Test flat sky
    sky = im.copy()
    skycountspersecond = 1
    sky.array[:] = skycountspersecond
    im2 = im.copy()
    l3.simulate_cps(im2, filter_name, exptime, sky=sky, seed=rng_seed,
                    etomjysr=1)
    # verify adding the sky increases the counts
    assert np.all(im2.array >= im.array)
    # verify that the count rate is about right.
    # poisson_rate = skycountspersecond * exptime
    poisson_rate = skycountspersecond
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
    assert (np.abs(poisson_rate - np.var(im2.array))
            < 10 * np.sqrt(var_of_var))

    # there are a few WCS bits where we use the positions in the catalog
    # to figure out where to render objects.  That would require setting
    # up a real PSF and is annoying.  Skipping that.
    # render some objects
    im3 = im.copy()
    _, objinfo = l3.simulate_cps(
        im3, filter_name, exptime, objlist=imdict['graycatalog'], psf=imdict['impsfgray'],
        xpos=[50, 50], ypos=[50, 50], seed=rng_seed)

    assert np.sum(im3.array) > 0  # at least verify that we added some sources...
    assert len(objinfo['simcatobj']) == 2  # two sources were added

    im4 = im.copy()
    _, objinfo = l3.simulate_cps(
        im4, filter_name, exptime, objlist=imdict['chromcatalog'],
        xpos=[50, 50], ypos=[50, 50],
        seed=rng_seed,
        psf=imdict['impsfchromatic'], bandpass=imdict['bandpass'])
    assert np.sum(im4.array) > 0  # at least verify that we added some sources...
    assert len(objinfo['simcatobj']) == 2  # two sources were added

    im5 = im.copy()
    _, objinfo = l3.simulate_cps(
        im5, filter_name, exptime, objlist=imdict['chromcatalog'],
        psf=imdict['impsfchromatic'], xpos=[1000, 1000],
        seed=rng_seed,
        ypos=[1000, 1000])
    assert 'simcatobj' not in objinfo
    # these sources should be out of bounds


def test_exptime_array():
    """ Test variable exposure times
    """
    # Set random seed
    rng_seed = 42

    # Set image, catalog, etc.
    imdict = set_up_image_rendering_things()
    im = imdict['im'].copy()
    im.array[:] = 0
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
    parameters.n_pix = 100

    # Create metadata
    meta = util.default_image_meta(filter_name='F158')
    wcs.fill_in_parameters(meta, coord)
    metadata = deepcopy(parameters.default_mosaic_parameters_dictionary)
    filter_name = 'F158'
    metadata['instrument']['optical_element'] = filter_name
    metadata['wcsinfo']['ra_ref'] = 270
    metadata['wcsinfo']['dec_ref'] = 66

    # Set variable exposure time array
    exptime = np.ones((parameters.n_pix, parameters.n_pix))
    basetime = 300
    expfactor = 2
    exptime[0:50, :] = basetime
    exptime[50:, :] = basetime * expfactor

    # Set WCS
    twcs = romanisim.models.wcs.get_mosaic_wcs(metadata, shape=(parameters.n_pix, parameters.n_pix))

    # Create chromatic data simulation
    im1, extras1 = l3.simulate((parameters.n_pix, parameters.n_pix), twcs, exptime, filter_name, chromcat,
                               bandpass=imdict['bandpass'], seed=rng_seed,
                               metadata=metadata, ignore_distant_sources=100,
                               effreadnoise=0,
                               )

    # Create filter data simulation
    im2, extras2 = l3.simulate((parameters.n_pix, parameters.n_pix), twcs, exptime, filter_name, graycat,
                               psf=imdict['impsfgray'],
                               seed=rng_seed, metadata=metadata,
                               ignore_distant_sources=100,
                               effreadnoise=0,
                               )

    # Ensure that the poisson variance scales with exposure time difference
    assert np.isclose((np.median(im1['var_poisson'][0:50, :]) / np.median(im1['var_poisson'][50:, :])), expfactor, rtol=0.02)
    assert np.isclose((np.median(im2['var_poisson'][0:50, :]) / np.median(im2['var_poisson'][50:, :])), expfactor, rtol=0.02)

    # Ensure that the data remains consistent across the exposure times
    assert np.isclose(np.median(im1['data'][0:50, :]), np.median(im1['data'][50:, :]), rtol=0.02)
    assert np.isclose(np.median(im2['data'][0:50, :]), np.median(im2['data'][50:, :]), rtol=0.02)


def test_scaling():
    npix = 200
    imdict = set_up_image_rendering_things()
    rng_seed = 1
    exptime = 400
    pscale = 0.1
    coord = SkyCoord(270 * u.deg, 66 * u.deg)

    # Set WCS
    twcs1 = romanisim.models.wcs.create_tangent_plane_gwcs(
        (npix / 2, npix / 2), pscale, coord)
    twcs2 = romanisim.models.wcs.create_tangent_plane_gwcs(
        (npix / 2, npix / 2), pscale / 2, coord)

    im1, extras1 = l3.simulate(
        (npix, npix), twcs1, exptime, imdict['filter_name'],
        imdict['tabcatalog'], seed=rng_seed, effreadnoise=0,
        )

    # half pixel scale
    im2, extras2 = l3.simulate(
        (npix * 2, npix * 2), twcs2, exptime, imdict['filter_name'],
        imdict['tabcatalog'], seed=rng_seed, effreadnoise=0)

    # check that sky level doesn't depend on pixel scale (in calibrated units!)
    skyfracdiff = np.median(im1.data) / np.median(im2.data) - 1
    log.info(f'skyfracdiff: {skyfracdiff:.3f}')
    assert np.abs(skyfracdiff) < 0.1

    # check that uncertainties match observed standard deviations
    err1fracdiff = mad_std(im1.data) / np.median(im1.err) - 1
    err2fracdiff = mad_std(im2.data) / np.median(im2.err) - 1
    log.info(f'err1fracdiff: {err1fracdiff:.3f}, '
             f'err2fracdiff: {err2fracdiff:.3f}')
    assert np.abs(err1fracdiff) < 0.1
    assert np.abs(err2fracdiff) < 0.1

    # doubled exposure time
    im3, extras3 = l3.simulate(
        (npix, npix), twcs1, exptime * 10, imdict['filter_name'],
        imdict['tabcatalog'], seed=rng_seed, effreadnoise=0)

    # check that sky level doesn't depend on exposure time (in calibrated units!)
    sky3fracdiff = np.median(im1.data) / np.median(im3.data) - 1
    log.info(f'sky3fracdiff: {sky3fracdiff:.4f}')
    assert np.abs(sky3fracdiff) < 0.1

    # check that variances still work out
    err3fracdiff = mad_std(im3.data) / np.median(im3.err) - 1
    log.info(f'err3fracdiff: {err3fracdiff:.3f}')
    assert np.abs(err3fracdiff) < 0.1

    # check that new variances are smaller than old ones by an appropriate factor
    errfracdiff = (
        np.median(im1.err) / np.median(im3.err) - np.sqrt(10))
    log.info(f'err3 ratio diff from 1/sqrt(10) err1: {errfracdiff:0.3f}')
    assert np.abs(errfracdiff) < 0.1

    # check that fluxes match
    # pixel scales are different by a factor of two.
    fluxes = []
    for im, fac in zip((im1, im2, im3), (1, 2, 1)):
        pix = im.meta.wcs.world_to_pixel_values(
            imdict['tabcatalog']['ra'][0], imdict['tabcatalog']['dec'][0])
        pind = [int(x) for x in pix]
        margin = 30 * fac
        flux = np.sum(im.data[pind[1] - margin: pind[1] + margin,
                                    pind[0] - margin: pind[0] + margin])
        fluxes.append(flux / fac ** 2)
        # division by fac ** 2 accounts for different pixel scale
        # i.e., we should be doing an integral here, and the pixels
        # are a factor of 4 smaller in the second integral
    # fluxes must match
    for flux in fluxes[1:]:
        fluxfracdiff = flux / fluxes[0] - 1
        log.info(f'fluxfracdiff: {fluxfracdiff:.5f}')
        assert np.abs(fluxfracdiff) < 0.1

    # these all match to a few percent; worst case in initial test run
    # was err3fracdiff of 0.039.
