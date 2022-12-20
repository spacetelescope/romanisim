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

import numpy as np
import galsim
from galsim import roman
from romanisim import image, parameters, catalog, psf
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import asdf


def test_in_bounds():
    bounds = galsim.BoundsI(0, 1000, 0, 1000)
    xx = np.random.rand(1000)*1000
    yy = np.random.rand(1000)*1000
    assert np.all(image.in_bounds(xx, yy, bounds, 0))
    xx = np.array([-1, 1001, -2, 1002, 50, 51])
    yy = np.array([50, 51, -20, 1002, -50, 1051])
    assert ~np.any(image.in_bounds(xx, yy, bounds, 0))
    assert np.all(image.in_bounds(xx, yy, bounds, 60))


def test_make_l2():
    resultants = np.ones((4, 50, 50), dtype='i4')
    ma_table = [[0, 4], [4, 4], [8, 4], [12, 4]]
    slopes, readvar, poissonvar = image.make_l2(
        resultants, ma_table, gain=1, flat=1, dark=0)
    assert np.allclose(slopes, 0)
    resultants[:, :, :] = np.arange(4)[:, None, None]
    slopes, readvar, poissonvar = image.make_l2(
        resultants, ma_table, gain=1, flat=1, dark=0)
    assert np.allclose(slopes,  1/parameters.read_time/4)
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
    print(readvar2)
    print(readvar1)
    print(readvar2/readvar1)
    assert np.all(np.abs(readvar2 / (readvar1 * 4) - 1) < 0.1)
    slopes2, readvar2, poissonvar2 = image.make_l2(
        resultants, ma_table, read_noise=1, flat=0.5)
    assert np.allclose(slopes2, slopes1 / 0.5)
    assert np.allclose(readvar2, readvar1 / 0.5**2)
    assert np.allclose(poissonvar2, poissonvar1 / 0.5**2)
    slopes, readvar, poissonvar = image.make_l2(
        resultants, ma_table, read_noise=1, gain=1, flat=1,
        dark=resultants)
    assert np.allclose(slopes, 0)


def set_up_image_rendering_things():
    im = galsim.Image(100, 100, scale=0.1)
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
    return dict(im=im, impsfgray=impsfgray,
                impsfchromatic=impsfchromatic,
                bandpass=bandpass, counts=counts, fluxdict=fluxdict,
                graycatalog=graycatalog,
                chromcatalog=chromcatalog, filter_name=filter_name)


def test_add_objects():
    imdict = set_up_image_rendering_things()
    im, impsfgray = imdict['im'], imdict['impsfgray']
    impsfchromatic = imdict['impsfchromatic']
    bandpass, counts = imdict['bandpass'], imdict['counts']
    fluxdict, graycatalog = imdict['fluxdict'], imdict['graycatalog']
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
    print(im.bounds)
    assert (peakloc[0] == 60) & (peakloc[1] == 30)


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
    assert (np.mean(im2.array) - zpflux * exptime <
            20 * np.sqrt(skycountspersecond * zpflux * exptime))
    im3 = im.copy()
    image.simulate_counts_generic(im3, exptime, dark=sky, zpflux=zpflux)
    # verify that the dark counts don't see the zero point conversion
    assert (np.mean(im3.array) - exptime <
            20 * np.sqrt(skycountspersecond * exptime))
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
    # def simulate_counts(sca, targ_pos, date, objlist, filter_name,
    #                     exptime=None, rng=None, seed=None,
    #                     ignore_distant_sources=10, usecrds=True,
    #                     return_info=False, webbpsf=True,
    #                     darkrate=None, flat=None):
    imdict = set_up_image_rendering_things()
    chromcat = imdict['chromcatalog']
    graycat = imdict['graycatalog']
    coord = SkyCoord(270 * u.deg, 66 * u.deg)
    time = Time('2020-01-01T00:00:00')
    for o in chromcat:
        o.sky_pos = coord
    for o in graycat:
        o.sky_pos = coord
    # these are all dumb coordinates; the coord sent to simulate_counts
    # is the coordinate of the boresight, but that doesn't need to be on SCA 1.
    # But at least they'll exercise some machinery if the ignore_distant_sources
    # argument is high enough!
    roman.n_pix = 100
    im1 = image.simulate_counts(1, coord, time, chromcat, 'F158',
                                usecrds=False, webbpsf=False,
                                ignore_distant_sources=100)
    im2 = image.simulate_counts(1, coord, time, graycat, 'F158',
                                usecrds=False, webbpsf=True,
                                ignore_distant_sources=100)
    print(im1[1])
    print(im2[1])
    im1 = im1[0].array
    im2 = im2[0].array
    maxim = np.where(im1 > im2, im1, im2)
    m = np.abs(im1 - im2) <= 20 * np.sqrt(maxim)
    print(np.sum(~m))
    print(im1[~m], im2[~m])
    assert np.all(m)


def test_simulate(tmp_path):
    imdict = set_up_image_rendering_things()
    # simulate gray, chromatic, level0, level1, level2 images
    roman.n_pix = 100
    meta = dict()
    coord = SkyCoord(270 * u.deg, 66 * u.deg)
    time = Time('2020-01-01T00:00:00')
    meta['roman.meta.exposure.start_time'] = time
    meta['roman.meta.pointing.ra_v1'] = coord.ra.to(u.deg).value
    meta['roman.meta.pointing.dec_v1'] = coord.dec.to(u.deg).value
    chromcat = imdict['chromcatalog']
    graycat = imdict['graycatalog']
    for o in chromcat:
        o.sky_pos = coord
    for o in graycat:
        o.sky_pos = coord
    l0 = image.simulate(meta, graycat, webbpsf=True, level=0, usecrds=False)
    l1 = image.simulate(meta, graycat, webbpsf=True, level=1, usecrds=False)
    l2 = image.simulate(meta, graycat, webbpsf=True, level=2, usecrds=False)
    l2c = image.simulate(meta, chromcat, webbpsf=False, level=2, usecrds=False)
    # what should we test here?  At least that the images validate?
    # I've already tested as many of the image generation things as I can
    # think of at earlier stages.
    assert isinstance(l0, galsim.Image)
    for ll in [l1, l2, l2c]:
        af = asdf.AsdfFile()
        af.tree = {'roman': ll[0]}
        af.validate()
    l2, cat = l2
    res = image.make_asdf(l2['data'], l2['var_rnoise'], l2['var_poisson'],
                          filepath=tmp_path / 'l2.asdf')
    af.tree = {'roman': res}
    af.validate()
    

def test_make_catalog_and_images():
    # this isn't a real routine that we should consider part of the
    # public interface, and may be removed.  We'll settle for just
    # testing that it runs.
    roman.n_pix = 100
    res = image.make_test_catalog_and_images(usecrds=False)
    assert len(res) > 0
