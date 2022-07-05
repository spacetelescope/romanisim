"""Roman WFI simulator tool.

Based on demo13.py in galsim.  Uses galsim Roman modules for all of the real
work.

Current design is centered around simulating a single SCA.  It will not handle
objects that cross SCAs.
"""

import sys
import datetime
from dataclasses import dataclass
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import galsim
from galsim import roman
import roman_datamodels.testing.utils

import logging
logging.basicConfig(format='%(message)s', stream=sys.stdout)
log = logging.getLogger()


@dataclass
class CatalogObject:
    """Simple class to hold galsim positions and profiles of objects."""
    image_pos: galsim.Position
    profile: galsim.GSObject


def make_dummy_catalog(rng=None, seed=42, nobj=1000):
    """Make a dummy catalog for testing purposes.

    Params
    ------
    rng : Galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating random number generator.  Only used if rng is None.
    nobj : int
        Number of objects to simulate.
    """
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area,
                                exptime=roman.exptime)
    cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area,
                                exptime=roman.exptime)

    # following Roman demo13, all stars currently have the SED of Vega.
    # fluxes are set to have a specific value in the y bandpass.
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
    y_bandpass = roman.getBandpasses(AB_zeropoint=True)['Y106']

    objlist = []
    for i in range(nobj):
        x = rng() * roman.n_pix
        y = rng() * roman.n_pix
        image_pos = galsim.PositionD(x, y)
        p = rng()
        # prescription follows galsim demo13.
        if p < 0.8:  # 80% of targets; faint galaxies
            obj = cat1.makeGalaxy(chromatic=True, gal_type='parametric',
                                  rng=rng)
            theta = rng() * 2 * np.pi * galsim.radians
            obj = obj.rotate(theta)
        elif p < 0.9:  # 10% of targets; stars
            mu_x = 1.e5
            sigma_x = 2.e5
            mu = np.log(mu_x**2 / (mu_x**2+sigma_x**2)**0.5)
            sigma = (np.log(1 + sigma_x**2/mu_x**2))**0.5
            gd = galsim.GaussianDeviate(rng, mean=mu, sigma=sigma)
            flux = np.exp(gd())
            sed = vega_sed.withFlux(flux, y_bandpass)
            obj = galsim.DeltaFunction() * sed
        else:  # 10% of targets; bright galaxies
            obj = cat2.makeGalaxy(chromatic=True, gal_type='parametric',
                                  rng=rng)
            obj = obj.dilate(2) * 4
            theta = rng() * 2 * np.pi * galsim.radians
            obj = obj.rotate(theta)
        objlist.append(CatalogObject(image_pos, obj))
    return objlist


def simulate_filter(sca, targ_pos, date, objlist, filter_name, exptime=None,
                    rng=None, seed=None, return_variance=False):
    """Simulate observations of a single SCA.

    Parameters
    ----------
    sca : int
        SCA to simulate
    targ_pos : galsim.CelestialCoord
        Location on sky to observe
    date : datetime
        Time at which to simulate observation
    objlist : list[CatalogObject]
        Objects to simulate
    filter_name : str
        Roman filter bandpass to use
    exptime : float
        Exposure time to use (if None, default to roman.exptime)
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.
    return_variance : bool
        If True, also return var_poisson, var_rnoise, var_flat
        Poisson noise currently does not think about instrumental
        effects like reciprocity failure, non-linearity, IPC.
    """
    if exptime is None:
        exptime = roman.exptime
    if rng is None and seed is None:
        seed = 43
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    bandpass = roman.getBandpasses(AB_zeropoint=True)[filter_name]
    wcs_dict = roman.getWCS(world_pos=targ_pos, SCAs=sca, date=date)
    wcs = wcs_dict[sca]
    psf = roman.getPSF(sca, filter_name, n_waves=10, wcs=wcs, pupil_bin=8)
    full_image = galsim.ImageF(roman.n_pix, roman.n_pix, wcs=wcs)
    sky_image = galsim.ImageF(roman.n_pix, roman.n_pix, wcs=wcs)

    SCA_cent_pos = wcs.toWorld(sky_image.true_center)
    sky_level = roman.getSkyLevel(bandpass, world_pos=SCA_cent_pos)
    sky_level *= (1.0 + roman.stray_light_fraction)
    wcs.makeSkyImage(sky_image, sky_level)
    sky_image += roman.thermal_backgrounds[filter_name] * roman.exptime

    for obj in objlist:
        final = galsim.Convolve(obj.profile, psf)
        stamp = final.drawImage(
            bandpass, center=obj.image_pos, wcs=wcs.local(obj.image_pos),
            method='phot', rng=rng)
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

    var_poisson = full_image + sky_image
    # note this is imperfect because the 'photon shooting' psf modeling
    # has poisson noise in it.

    full_image.quantize()

    poisson_noise = galsim.PoissonNoise(rng)
    sky_image.addNoise(poisson_noise)
    full_image += sky_image

    roman.addReciprocityFailure(full_image)
    dark_current = roman.dark_current * roman.exptime
    dark_noise = galsim.DeviateNoise(
        galsim.PoissonDeviate(rng, dark_current))
    var_poisson += dark_current
    full_image.addNoise(dark_noise)
    roman.applyNonlinearity(full_image)
    roman.applyIPC(full_image)
    read_noise = galsim.GaussianNoise(rng, sigma=roman.read_noise)
    full_image.addNoise(read_noise)

    full_image /= roman.gain
    full_image.quantize()

    if return_variance:
        var_rnoise = full_image*0 + roman.read_noise**2
        var_flat = full_image*0
        full_image = (full_image, var_poisson, var_rnoise, var_flat) 
    
    return full_image


def simulate(coord, date, objlist, sca, filters, seed=12345, exptime=None,
             return_variance=False):
    """Simulate a sequence of observations on a field in different bandpasses.

    coord : astropy.coordinates.SkyCoord
        Sky location to simulation
    date : datetime.datetime
        The time of the observation
    objlist : list[CatalogObject]
        List of objects in the field to simulate
    sca : int
        SCA to simulate
    filters : list[str]
        filters to use, e.g., ['Z087', 'Y106', 'J129']
    return_variance : bool
        if True, also include var_poisson, var_rnoise, var_flat parallel to
        the images in the dictionary

    Returns
    -------
    dict[str] -> galsim.Image
        dictionary giving simulated images for each filter

    If return_variance is True, then instead
    dict[str] -> (image, var_poisson, var_rnoise, var_flat)
        where each of image, var_poisson, var_rnoise, var_flat are galsim.Image
        and give the image value and its Poisson, read noise, and flat field
        induced variances.
    """
    if exptime is None:
        exptime = roman.exptime

    ra_targ = galsim.Angle(coord.ra.to(u.rad).value, galsim.angle.radians)
    dec_targ = galsim.Angle(coord.dec.to(u.rad).value, galsim.angle.radians)
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)

    out = dict()

    for i, filter_name in enumerate(filters):
        log.info('Simulating filter {0}...'.format(filter_name))
        out[filter_name] = simulate_filter(
            sca, targ_pos, date, objlist, filter_name, seed=i+1+seed,
            exptime=exptime, return_variance=return_variance)
    return out


def make_test_catalog_and_images(seed=12345, sca=7, filters=None, nobj=1000,
                                 return_variance=False):
    """This routine kicks the tires on everything in this module."""
    log.info('Making catalog...')
    if filters is None:
        filters = ['Y106', 'J129', 'H158']
    cat = make_dummy_catalog(seed=seed, nobj=nobj)
    # ecliptic pole is probably always visible?
    coord = SkyCoord(ra=270*u.deg, dec=66*u.deg)
    date = datetime.datetime(2027, 1, 1)
    return simulate(coord, date, cat, sca, filters, seed=seed,
                    return_variance=return_variance)


def galsim_to_asdf(im, var_poisson, var_rnoise, var_flat, sca, bandpass):
    """Wrap a galsim simulated image with ASDF/roman_datamodel metadata."""

    out = roman_datamodels.testing.utils.mk_level2_image()
    # fill this output with as much real information as possible.
    # aperture['name'] gets the correct SCA
    # aperture['position_angle'] gets the correct PA
    # cal['step'] is left empty for now, but in principle
    #     could be filled out at some level
    # ephemeris contains a lot of angles that could be computed.
    # exposure contains start_time, mid_time, end_time
    #     plus MJD equivalents
    #     plus TDB equivalents (?)
    #     plus ENG equivalents (?)
    #     ngroups, nframes, sca_number, gain_factor, integration_time,
    #     elapsed_exposure_time, nints, integration_start, integration_end,
    #     frame_divisor, groupgap, nsamples, sample_time, frame_time,
    #     group_time, exposure_time, effective_exposure_time,
    #     duration, nresets_at_start, datamode, ma_atble_name, ma_table_number,
    # observation: some of this could be passed forward from the
    #     APT file.  e.g., program, pass, observation_label,
    # pointing
    # program: title, pi_name, category, subcategory, science_category,
    #     available in APT file
    # ref_file: conceptually sound when we work from crds reference
    #     files
    # target: can push forward from APT file
    # velocity_aberration:  I guess to first order,
    #     ignore the detailed orbit around L2 and just project
    #     the earth out to L2, and use that for the velocity
    #     aberration?  Don't do until someone asks.
    # visit: start_time, end_time, total_exposures, ...?
    # wcsinfo: v2_ref, v3_ref, vparitiy, v3yangle, ra_ref, dec_ref
    #     roll_ref, s_region
    # photometry: all of these conversions could be filled out
    #     just requires the assumed zero point, some definitions,
    #     and the WCS.
    # border_ref_pix_left, _right, _top, _bottom:
    #     leave reference blank for now
    # dq_border_pix_left, _right, _top, _bottom:
    #     leave border pixels blank for now.
    # amp33: leave blank for now.
    # data: image
    # dq: I guess all 0?
    # err: I guess I need to look up how this is defined.  But
    #     the "right" thing is probably
    #     sqrt(noisless image + read_noise**2)
    #     i.e., Gaussian approximation of Poisson noise + read noise.
    #     this is just sqrt(var_poisson + var_rnoise + var_flat)?
    # var_rnoise: okay
    # var_flat: currently zero
    out['meta']['aperture']['name'] = 'WFI_{}_FULL'.format(sca)
    out['meta']['instrument']['detector'] = 'WFI{}'.format(sca)
    out['meta']['instrument']['optical_element'] = bandpass
    out['data'] = im.array
    out['dq'] = 0
    out['var_poisson'] = var_poisson.array
    out['var_rnoise'] = var_rnoise.array
    out['var_flat'] = var_flat.array
    out['err'] = np.sqrt(var_poisson.array + var_rnoise.array + var_flat.array)
    return out
