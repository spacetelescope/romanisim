"""Unit tests for image module.

These routines exercise the following:
- make_l2: resultants -> l2, eventually belongs in an l2 module
- in_bounds: check if points are in bounds
- add_objects_to_image: Adds some objects to an image.
- simulate_counts_generic: Adds sky, dark, and an object list to an image.
- simulate_counts: Wraps simulate_counts_generic, making sky & dark images
  for Roman to pass on
- simulate: reads in Roman calibration files.  Sends to simulate_counts, and
  further passes that to L1 and L2 routines.
- make_test_catalog_and_images: routine which kicks the tires on everything.
- make_asdf: Makes an l2 asdf file.
"""

import os
import copy
import numpy as np
import galsim
from galsim import roman
from romanisim import image, parameters, catalog, psf, util, wcs, persistence
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy import table
import asdf
import webbpsf
from astropy.modeling.functional_models import Sersic2D
import pytest
from romanisim import log
from roman_datamodels.stnode import WfiScienceRaw, WfiImage


def test_in_bounds():
    bounds = galsim.BoundsI(0, 1000, 0, 1000)
    xx = np.random.rand(1000) * 1000
    yy = np.random.rand(1000) * 1000
    assert np.all(image.in_bounds(xx, yy, bounds, 0))
    xx = np.array([-1, 1001, -2, 1002, 50, 51])
    yy = np.array([50, 51, -20, 1002, -50, 1051])
    assert ~np.any(image.in_bounds(xx, yy, bounds, 0))
    assert np.all(image.in_bounds(xx, yy, bounds, 60))


def test_trim_objlist():
    cen = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
    cat1 = catalog.make_dummy_table_catalog(cen, radius=0.001, nobj=1000)
    # sources within 0.1 deg
    cen2 = SkyCoord(ra=180 * u.deg, dec=0 * u.deg)
    cat2 = catalog.make_dummy_table_catalog(cen2, radius=0.001, nobj=1000)
    cat3 = catalog.make_dummy_table_catalog(cen, radius=0.003, nobj=1000)
    affine = galsim.AffineTransform(
        0.1, 0, 0, 0.1, origin=galsim.PositionI(50, 50),
        world_origin=galsim.PositionD(0, 0))
    wcs = galsim.TanWCS(affine,
                        util.celestialcoord(cen))
    # image is ~50 pix = 5 arcsec ~ 0.0013 deg.
    im = galsim.Image(100, 100, wcs=wcs)

    outcat = image.trim_objlist(cat1, im)
    assert len(outcat) == len(cat1)  # no objects trimmed
    outcat = image.trim_objlist(cat2, im)
    assert len(outcat) == 0  # all objects trimmed
    outcat = image.trim_objlist(cat3, im)
    assert (len(outcat) > 0) and (len(outcat) < len(cat3))
    # some objects in, some objects out


def test_make_l2():
    resultants = np.ones((4, 50, 50), dtype='i4')
    ma_table = [[0, 4], [4, 4], [8, 4], [12, 4]]
    slopes, readvar, poissonvar = image.make_l2(
        resultants, ma_table, gain=1, flat=1, dark=0)
    assert np.allclose(slopes, 0)
    resultants[:, :, :] = np.arange(4)[:, None, None]
    resultants *= u.DN
    gain = 1 * u.electron / u.DN
    slopes, readvar, poissonvar = image.make_l2(
        resultants, ma_table,
        gain=gain, flat=1, dark=0)
    assert np.allclose(slopes, 1 / parameters.read_time / 4 * u.electron / u.s)
    assert np.all(np.array(slopes.shape) == np.array(readvar.shape))
    assert np.all(np.array(slopes.shape) == np.array(poissonvar.shape))
    assert np.all(readvar >= 0)
    assert np.all(poissonvar >= 0)
    slopes1, readvar1, poissonvar1 = image.make_l2(
        resultants, ma_table, read_noise=1, dark=0)
    slopes2, readvar2, poissonvar2 = image.make_l2(
        resultants, ma_table, read_noise=2, dark=0)
    assert np.all(readvar2 >= readvar1)
    # because we change the weights depending on the ratio of read & poisson
    # noise, we can't assume above that readvar2 = readvar1 * 4.
    # But it should be pretty darn close here.
    assert np.all(np.abs(readvar2 / (readvar1 * 4) - 1) < 0.1)
    slopes2, readvar2, poissonvar2 = image.make_l2(
        resultants, ma_table, read_noise=1, flat=0.5)
    assert np.allclose(slopes2, slopes1 / 0.5)
    assert np.allclose(readvar2, readvar1 / 0.5**2)
    assert np.allclose(poissonvar2, poissonvar1 / 0.5**2)
    slopes, readvar, poissonvar = image.make_l2(
        resultants, ma_table, read_noise=1, gain=gain, flat=1,
        dark=resultants)
    assert np.allclose(slopes, 0)


def set_up_image_rendering_things():
    im = galsim.Image(100, 100, scale=0.1, xmin=0, ymin=0)
    filter_name = 'F158'
    impsfgray = psf.make_psf(1, filter_name, webbpsf=True, chromatic=False)
    impsfchromatic = psf.make_psf(1, filter_name, webbpsf=False, chromatic=True)
    bandpass = roman.getBandpasses(AB_zeropoint=True)['H158']
    counts = 1000
    fluxdict = {filter_name: counts}
    graycatalog = [
        catalog.CatalogObject(None, galsim.DeltaFunction(), fluxdict),
        catalog.CatalogObject(None, galsim.Sersic(4, half_light_radius=1),
                              fluxdict)
    ]
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
    vega_sed = vega_sed.withFlux(counts, bandpass)
    chromcatalog = [
        catalog.CatalogObject(None, galsim.DeltaFunction() * vega_sed, None),
        catalog.CatalogObject(
            None, galsim.Sersic(4, half_light_radius=1) * vega_sed, None)
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


def central_stamp(im, sz):
    for s in im.shape:
        if (s % 2) != 1:
            raise ValueError('size must be odd')
    if (sz % 2) != 1:
        raise ValueError('sz must be odd')
    center = im.shape[0] // 2
    szo2 = sz // 2
    return im[center - szo2: center + szo2 + 1,
              center - szo2: center + szo2 + 1]


@pytest.mark.soctests
def test_image_rendering():
    """Tests for image rendering routines.  This is demonstrates:
    - RUSBREQ-830 / DMS214: point source generation.
    - RSUBREQ-874 / DMS215: analytic model source generation
    """
    oversample = 4
    filter_name = 'F158'
    sca = 1
    pos = [50, 50]
    impsfgray = psf.make_psf(sca, filter_name, webbpsf=True, chromatic=False,
                             pix=pos, oversample=oversample)
    counts = 100000
    fluxdict = {filter_name: counts}
    psfcatalog = [catalog.CatalogObject(None, galsim.DeltaFunction(), fluxdict)]
    wfi = webbpsf.WFI()
    wfi.detector = f'SCA{sca:02d}'
    wfi.filter = filter_name
    wfi.detector_position = pos
    # oversample = kw.get('oversample', 4)
    # webbpsf doesn't do distortion
    psfob = wfi.calc_psf(oversample=oversample)
    psfim = psfob[1].data * counts
    # PSF from WebbPSF
    im = galsim.Image(101, 101, scale=wfi.pixelscale, xmin=0, ymin=0)
    # also no distortion
    image.add_objects_to_image(im, psfcatalog, [pos[0]], [pos[1]],
                               impsfgray, flux_to_counts_factor=1,
                               filter_name=filter_name)
    # im has psf from romanisim.  psfim has psf from webbpsf.  Now compare?
    # some notes: galsim is really trying to integrate over each pixel;
    # webbpsf is sampling at each 4x4 oversampled pixel and summing.
    # otherwise we expect perfection in the limit that the oversampling
    # in both make_psf and wfi.calc_psf becomes arbitrarily large?
    # object rendering also includes shot noise.
    sz = 5
    cenpsfim = central_stamp(psfim, sz)
    cenim = central_stamp(im.array, sz)
    uncertainty = np.sqrt(psfim)
    cenunc = central_stamp(uncertainty, sz)

    # DMS214 - our PSF matches that from WebbPSF
    assert np.max(np.abs((cenim - cenpsfim) / cenunc)) < 5
    log.info('DMS214: rendered PSF matches WebbPSF')

    # 5 sigma isn't so bad.
    # largest difference is in the center pixel, ~1.5%.  It seems to me that
    # this is from the simple oversampling-based integration; galsim should
    # be doing better here with a real integration.  Way out in the wings there
    # are some other artifacts at the 1% level that appear to be from the galsim
    # FFTs, though there are only a handful of counts out there.

    # DMS 215 - add a galaxy to an image
    # this makes a relatively easy sersic galaxy (index = 2)
    # and oversamples it very aggressively (32x).
    imsz = 101
    im = galsim.Image(imsz, imsz, scale=wfi.pixelscale, xmin=0, ymin=0)
    sersic_index = 2
    sersic_hlr = 0.6
    sersic_profile = galsim.Sersic(sersic_index, sersic_hlr)
    galcatalog = [catalog.CatalogObject(None, sersic_profile, fluxdict)]
    # impsfgray2 = galsim.DeltaFunction()
    image.add_objects_to_image(im, galcatalog, [pos[0]], [pos[1]], impsfgray,
                               flux_to_counts_factor=1, filter_name=filter_name)
    # now we have an image with a galaxy in it.
    # how else would we make that galaxy?
    # integral of a sersic profile:
    # Ie * re**2 * 2 * pi * n * exp(bn)/(bn**(2*n)) * g2n
    from scipy.special import gamma, gammaincinv
    oversample = 15
    # oversample must be odd, because we need the fftconvolve
    # kernel image size to be odd for mode='same' not to do an image shift
    # by one pixel.
    off = (oversample - 1) / 2
    xx, yy = np.meshgrid(
        np.arange(imsz * oversample), np.arange(imsz * oversample))
    mod = Sersic2D(amplitude=1, r_eff=oversample * sersic_hlr / wfi.pixelscale,
                   n=sersic_index,
                   x_0=pos[0] * oversample + off, y_0=pos[0] * oversample + off,
                   ellip=0, theta=0)
    modim = mod(xx, yy)
    from scipy.signal import fftconvolve
    # convolve with the PSF
    psfob = wfi.calc_psf(oversample=oversample, fov_pixels=45)
    psfim = psfob[0].data
    modim = fftconvolve(modim, psfim, mode='same')
    modim = np.sum(modim.reshape(-1, imsz, oversample), axis=-1)
    modim = np.sum(modim.reshape(imsz, oversample, -1), axis=1)
    bn = gammaincinv(2 * sersic_index, 0.5)
    g2n = gamma(2 * sersic_index)
    integral = (1 * (oversample * sersic_hlr / wfi.pixelscale)**2 * g2n * 2
                * np.pi * sersic_index * np.exp(bn) / bn**(2 * sersic_index))
    assert abs(np.sum(modim) / integral - 1) < 0.05
    # if we did this right, we should have counted at least 95% of the flux.
    # note that the PSF convolution loses 3.2% outside its aperture!
    # we could use a larger PSF, but fov_pixels = 45 corresponds to the
    # default used above.
    modim = modim * counts / integral
    # now we need to compare the middles of the images
    # let's say inner 11x11 pixels
    sz = 11
    cenmodim = central_stamp(modim, sz)
    cenim = central_stamp(im.array, sz)
    cenunc = np.sqrt(cenim)

    # DMS 215
    assert np.max(np.abs((cenim - cenmodim) / cenunc)) < 5
    log.info('DMS215: rendered galaxy matches astropy Sersic2D after '
             'pixel integration.')

    # our two different realizations of this PSF convolved model
    # Sersic galaxy agree at <5 sigma on all pixels using only
    # Poisson errors and containing 100k counts.


def test_add_objects():
    imdict = set_up_image_rendering_things()
    im, impsfgray = imdict['im'], imdict['impsfgray']
    impsfchromatic = imdict['impsfchromatic']
    bandpass, counts = imdict['bandpass'], imdict['counts']
    graycatalog = imdict['graycatalog']
    chromcatalog = imdict['chromcatalog']
    image.add_objects_to_image(im, graycatalog, [50, 50], [50, 50],
                               impsfgray, flux_to_counts_factor=1,
                               filter_name=imdict['filter_name'])
    assert (np.abs(np.sum(im.array) - 2 * counts) < 20 * np.sqrt(counts))
    im.array[:] = 0
    image.add_objects_to_image(im, chromcatalog, [60, 60], [30, 30],
                               impsfchromatic, flux_to_counts_factor=1,
                               bandpass=bandpass)
    assert (np.abs(np.sum(im.array) - 2 * counts) < 20 * np.sqrt(counts))
    peaklocs = np.where(im.array == np.max(im.array))
    peakloc = peaklocs[1][0] + im.bounds.xmin, peaklocs[0][0] + im.bounds.ymin
    assert (peakloc[0] == 60) & (peakloc[1] == 30)
    im.array[:] = 0
    image.add_objects_to_image(im, chromcatalog, [60, 60], [30, 30],
                               impsfchromatic, flux_to_counts_factor=2,
                               bandpass=bandpass)
    assert (np.abs(np.sum(im.array) - 2 * counts * 2) < 40 * np.sqrt(counts))


def test_simulate_counts_generic():
    imdict = set_up_image_rendering_things()
    im = imdict['im']
    im.array[:] = 0
    exptime = 100
    zpflux = 10
    image.simulate_counts_generic(
        im, exptime, objlist=[], psf=imdict['impsfgray'],
        sky=None, dark=None, flat=None, xpos=[], ypos=[],
        bandpass=None, filter_name=None, zpflux=zpflux)
    assert np.all(im.array == 0)  # verify nothing in -> nothing out
    sky = im.copy()
    skycountspersecond = 1
    sky.array[:] = skycountspersecond
    im2 = im.copy()
    image.simulate_counts_generic(im2, exptime, sky=sky, zpflux=zpflux)
    # verify adding the sky increases the counts
    assert np.all(im2.array >= im.array)
    # verify that the count rate is about right.
    assert (np.mean(im2.array) - zpflux * exptime
            < 20 * np.sqrt(skycountspersecond * zpflux * exptime))
    im3 = im.copy()
    image.simulate_counts_generic(im3, exptime, dark=sky, zpflux=zpflux)
    # verify that the dark counts don't see the zero point conversion
    assert (np.mean(im3.array) - exptime
            < 20 * np.sqrt(skycountspersecond * exptime))
    im4 = im.copy()
    image.simulate_counts_generic(im4, exptime, dark=sky, flat=0.5,
                                  zpflux=zpflux)
    # verify that dark electrons are not hit by the flat
    assert np.all(im3.array - im4.array
                  < 20 * np.sqrt(exptime * skycountspersecond))
    im5 = im.copy()
    image.simulate_counts_generic(im5, exptime, sky=sky, flat=0.5,
                                  zpflux=zpflux)
    # verify that sky photons are hit by the flat
    assert (np.mean(im5.array) - skycountspersecond * zpflux * exptime * 0.5
            < 20 * np.sqrt(skycountspersecond * zpflux * exptime * 0.5))
    # there are a few WCS bits where we use the positions in the catalog
    # to figure out where to render objects.  That would require setting
    # up a real PSF and is annoying.  Skipping that.
    # render some objects
    image.simulate_counts_generic(
        im, exptime, objlist=imdict['graycatalog'], psf=imdict['impsfgray'],
        xpos=[50, 50], ypos=[50, 50], zpflux=zpflux,
        filter_name=imdict['filter_name'])
    objinfo = image.simulate_counts_generic(
        im, exptime, objlist=imdict['chromcatalog'],
        xpos=[50, 50], ypos=[50, 50],
        psf=imdict['impsfchromatic'], bandpass=imdict['bandpass'])
    assert np.sum(im.array) > 0  # at least verify that we added some sources...
    assert len(objinfo) == 2  # two sources were added
    objinfo = image.simulate_counts_generic(
        im, exptime, objlist=imdict['chromcatalog'],
        psf=imdict['impsfchromatic'], xpos=[1000, 1000],
        ypos=[1000, 1000], zpflux=zpflux)
    assert np.sum(objinfo['counts'] > 0) == 0
    # these sources should be out of bounds


def test_simulate_counts():
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

    meta = {
        'exposure': {
            'start_time': Time('2020-01-01T00:00:00'),
            'ma_table_number' : 1,
        },
        'instrument': {
            'detector': 'WFI01',
            'optical_element': 'F158',
        },
    }
    wcs.fill_in_parameters(meta, coord)
    im1 = image.simulate_counts(meta, chromcat, usecrds=False,
                                webbpsf=False, ignore_distant_sources=100)
    im2 = image.simulate_counts(meta, graycat,
                                usecrds=False, webbpsf=True,
                                ignore_distant_sources=100)
    im1 = im1[0].array
    im2 = im2[0].array
    maxim = np.where(im1 > im2, im1, im2)
    m = np.abs(im1 - im2) <= 20 * np.sqrt(maxim)
    assert np.all(m)


@pytest.mark.soctests
def test_simulate(tmp_path):
    """Test convolved image generation and L2 simulation framework.
    Demonstrates DMS216: convolved image generation - Level 2
    Demonstrates DMS224: persistence.
    """
    imdict = set_up_image_rendering_things()
    # simulate gray, chromatic, level0, level1, level2 images
    roman.n_pix = 100
    coord = SkyCoord(270 * u.deg, 66 * u.deg)
    time = Time('2020-01-01T00:00:00')

    meta = {
        'exposure' : {
            'start_time' : time,
            'ma_table_number' : 1,
        },
        'instrument' : {
            'optical_element' : 'F158',
            'detector' : 'WFI01'
        },
        'pointing' : {
            'ra_v1' : coord.ra.to(u.deg).value,
            'dec_v1' : coord.dec.to(u.deg).value,
        },
    }

    chromcat = imdict['chromcatalog']
    graycat = imdict['graycatalog']
    for o in chromcat:
        o.sky_pos = coord
    for o in graycat:
        o.sky_pos = coord
    l0 = image.simulate(meta, graycat, webbpsf=True, level=0,
                        usecrds=False)
    l0tab = image.simulate(
        meta, imdict['tabcatalog'], webbpsf=True, level=0, usecrds=False)
    # seed = 0 is special and means "don't actually use a seed."  Any other
    # choice of seed gives deterministic behavior
    # note that we have scaled down the size of the image to 100x100 pix
    # in order to save time.  But the CR module fixes the area of the detector
    # rather than the area of a pixel, so this means that all of the normal
    # CRs are detected, except in a 100x100 region instead of a 4kx4k region;
    # i.e., there are 1600x too many CRs.  Fine for unit tests?
    rng = galsim.BaseDeviate(1)
    l1 = image.simulate(meta, graycat, webbpsf=True, level=1,
                        crparam=dict(), usecrds=False, rng=rng)
    rng = galsim.BaseDeviate(1)
    l1_nocr = image.simulate(meta, graycat, webbpsf=True, level=1,
                             usecrds=False, crparam=None, rng=rng)
    assert np.all(l1[0].data >= l1_nocr[0].data)
    log.info('DMS221: Successfully added cosmic rays to an L1 image.')
    l2 = image.simulate(meta, graycat, webbpsf=True, level=2,
                        usecrds=False, crparam=dict())
    # throw in some CRs for fun
    l2c = image.simulate(meta, chromcat, webbpsf=False, level=2,
                         usecrds=False)
    persist = persistence.Persistence()
    fluence = 30000
    persist.update(l0[0]['data'] * 0 + fluence, time.mjd - 100 / 60 / 60 / 24)
    # zap the whole frame, 100 seconds ago.
    rng = galsim.BaseDeviate(1)
    l1p = image.simulate(meta, graycat, webbpsf=True, level=1, usecrds=False,
                         persistence=persist, crparam=None, rng=rng)
    # the random number gets instatiated from the same seed, but the order in
    # which the numbers are generated is different so we can't guarantee, e.g.,
    # that all of the new values are strictly greater than the old ones.
    # But we've basically added a constant to the whole scene: we can at least
    # verify it's positive.
    diff = l1p[0]['data'][-1] * 1.0 - l1_nocr[0]['data'][-1] * 1.0
    # L1 files are unsigned, so the difference gets wonky unless you convert
    # to floats.
    # do we have a very rough guess for what the persistence should be?
    # note that getting this exactly right is hard unless we think about
    # how the persistence decays over the exposure
    roughguess = persistence.fermi(
        fluence, 170, parameters.persistence['A'],
        parameters.persistence['x0'], parameters.persistence['dx'],
        parameters.persistence['alpha'], parameters.persistence['gamma'])
    roughguess = roughguess * 140  # seconds of integration
    assert np.abs(
        np.log(np.mean(diff * parameters.gain).value / roughguess)) < 1
    # within a factor of e
    log.info('DMS224: added persistence to an image.')

    # what should we test here?  At least that the images validate?
    # I've already tested as many of the image generation things as I can
    # think of at earlier stages.
    assert isinstance(l0[0]['data'], np.ndarray)
    assert isinstance(l0tab[0]['data'], np.ndarray)
    for ll in [l1, l2, l2c]:
        af = asdf.AsdfFile()
        af.tree = {'roman': ll[0]}
        af.validate()
    l2, cat = l2
    res = image.make_asdf(l2['data'], l2['var_rnoise'], l2['var_poisson'],
                          filepath=tmp_path / 'l2.asdf')
    af.tree = {'roman': res}
    # DMS216
    af.validate()
    log.info('DMS216: successfully made a L2 file.')


def test_make_test_catalog_and_images():
    # this isn't a real routine that we should consider part of the
    # public interface, and may be removed.  We'll settle for just
    # testing that it runs.
    roman.n_pix = 100
    fn = os.environ.get('GALSIM_CAT_PATH', None)
    if fn is not None:
        fn = str(fn)
    res = image.make_test_catalog_and_images(usecrds=False,
                                             galaxy_sample_file_name=fn)
    assert len(res) > 0

@pytest.mark.parametrize(
    "level",
    [
        1, 2,
    ],
)
@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason=(
        "Roman CRDS servers are not currently available outside the internal network"
    ),
)
def test_reference_file_crds_match(level):
    # Set up parameters for simulation run
    galsim.roman.n_pix = 4088
    metadata = copy.deepcopy(parameters.default_parameters_dictionary)
    metadata['instrument']['detector'] = 'WFI07'
    metadata['instrument']['optical_element'] = 'F158'
    metadata['exposure']['ma_table_number'] = 1

    twcs = wcs.get_wcs(metadata, usecrds=True)
    rd_sca = twcs.toWorld(galsim.PositionD(
        galsim.roman.n_pix / 2, galsim.roman.n_pix / 2))

    cat = catalog.make_dummy_table_catalog(
        rd_sca, bandpasses=[metadata['instrument']['optical_element']], nobj=1000)

    rng = galsim.UniformDeviate(None)
    im, simcatobj = image.simulate(
        metadata, cat, usecrds=True,
        webbpsf=True, level=level,
        rng=rng)

    # Confirm that CRDS keyword was updated
    assert im.meta.ref_file.crds.sw_version != '12.3.1'

    if (level==1):
        assert (type(im) == WfiScienceRaw)
    else: #level = 2
        assert (type(im) == WfiImage)
