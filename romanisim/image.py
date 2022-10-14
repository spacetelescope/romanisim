"""Roman WFI simulator tool.

Based on galsim's implementation of Roman image simulation.  Uses galsim Roman modules
for most of the real work.
"""

import time
import logging
import copy
import numpy as np
import astropy.time
from astropy import units as u
from astropy import coordinates
import asdf
import galsim
from galsim import roman
import roman_datamodels.testing.utils
from . import wcs
from . import catalog
from . import bandpass
from . import psf
from . import parameters
from . import util
import romanisim.l1
import romanisim.bandpass
import romanisim.psf
from romanisim import log
from romanisim import parameters
import crds

# galsim fluxes are in photons / cm^2 / s
# we need to specify the area and exposure time in drawImage if
# specifying fluxes in physical units.
# presently we're behaving a bit badly in that the fluxes are given in
# physical units, but the galsim objects with SEDs are already scaled
# for Roman's collecting area and exposure time.
# We should divide those out so that we are always working in physical units
# and must include the area and exposure time in drawImage.
# I think that's just a division in make_dummy_catalog.
# Really, just setting exptime = area = 1 in COSMOSCatalog.
# then we also need to convert fluxes in maggies to fluxes in photons/cm^2/s.
# this is just some constant.

# it would be nice to not crash and burn when the rendering of a challenging
# object is requested---it's easy to make ~impossible to render Sersic galaxies
# with not obviously crazy sizes, indices, and minor/major ratios.
# So far, it seems the best parameter to control this is folding_ratio; I could
# imagine setting folding_ratio = 1e-2 without feeling too bad.
# it looks like this would need to be set both for the PSF profile and for the
# galaxy profile.
# I've at least caught these for the moment, but we should explore lower
# folding_ratio for speed, as well as directly rendering into the image.


def make_l2(resultants, ma_table, read_noise=None):
    """
    Simulate an image in a filter given resultants.

    This routine does idealized ramp fitting on a set of resultants.

    Parameters
    ----------
    resultants : np.ndarray[resultants, nx, ny]
        resultants array
    ma_table : list[list] (int)
        list of list of first read numbers and number of reads in each resultant
    read_noise : ndarray-like[n_pix, n_pix]
        read_noise image to use.  If None, use galsim.roman.read_noise.

    Returns
    -------
    im : galsim.Image
        best fitting slopes
    var_rnoise : galsim.Image
        variance in slopes from read noise
    var_poisson : galsim.Image
        variance in slopes from source noise
    """

    if read_noise is None:
        read_noise = galsim.roman.read_noise

    from . import ramp
    rampfitter = ramp.RampFitInterpolator(ma_table)
    log.warning('Gain should be handled as something more interesting '
                'than a single constant.')
    ramppar, rampvar = rampfitter.fit_ramps(resultants/galsim.roman.gain,
                                            read_noise)
    # could iterate if we wanted to improve the flux estimates

    return (ramppar[..., 1],  # the slopes, ignoring the pedestals
            rampvar[..., 0, 1, 1], # the read noise induced slope variance
            rampvar[..., 1, 1, 1], # the poisson noise induced slope variance
           )


def simulate_counts(sca, targ_pos, date, objlist, filter_name,
                    exptime=None, rng=None, seed=None,
                    ignore_distant_sources=10, usecrds=True,
                    return_info=False, webbpsf=True,
                    darkrate=None):
    """Simulate total counts in a single SCA.

    This gives the total counts in an idealized instrument with no systematics;
    it includes only distortion & PSF convolution.

    Parameters
    ----------
    sca : int
        SCA to simulate
    targ_pos : galsim.CelestialCoord or astropy.coordinates.SkyCoord
        Location on sky to observe; telescope boresight
    date : astropy.time.Time
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
    ignore_distant_sources : float
        do not render sources more than this many pixels off edge of detector
    usecrds : bool
        use CRDS distortion map
    darkrate : float or np.ndarray of float
        dark rate image to use (electrons / s)

    Returns
    -------
    galsim.Image
        idealized image of scene as seen by Roman, giving total electron counts
        from rate sources (astronomical objects; backgrounds; dark current) in
        each pixel.
    """
    if exptime is None:
        exptime = roman.exptime
    if rng is None and seed is None:
        seed = 43
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter_name]
    bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]
    imwcs = wcs.get_wcs(world_pos=targ_pos, date=date, sca=sca, usecrds=usecrds)
    chromatic = False
    if len(objlist) > 0 and objlist[0].profile.spectral:
        chromatic = True
    psf = romanisim.psf.make_psf(sca, filter_name, wcs=imwcs, chromatic=chromatic,
                                 webbpsf=webbpsf)
    full_image = galsim.ImageF(roman.n_pix, roman.n_pix, wcs=imwcs)
    sky_image = galsim.ImageF(roman.n_pix, roman.n_pix, wcs=imwcs)
    dark_image = galsim.ImageF(roman.n_pix, roman.n_pix, wcs=imwcs)

    SCA_cent_pos = imwcs.toWorld(sky_image.true_center)
    sky_level = roman.getSkyLevel(bandpass, world_pos=SCA_cent_pos,
                                  date=date.datetime, exptime=1) * exptime
    sky_level *= (1.0 + roman.stray_light_fraction)
    imwcs.makeSkyImage(sky_image, sky_level)
    sky_image += roman.thermal_backgrounds[galsim_filter_name] * exptime
    imbd = full_image.bounds
    abflux = romanisim.bandpass.get_abflux(filter_name)
    log.info('Adding sources to image...')

    nrender = 0
    final = None
    info = []
    for i, obj in enumerate(objlist):
        # this is kind of slow.  We need to do an initial vectorized cull before
        # reaching this point.
        t0 = time.time()
        image_pos = imwcs.toImage(obj.sky_pos)
        if ((image_pos.x < imbd.xmin - ignore_distant_sources) or
                (image_pos.x > imbd.xmax + ignore_distant_sources) or
                (image_pos.y < imbd.ymin - ignore_distant_sources) or
                (image_pos.y > imbd.ymax + ignore_distant_sources)):
            # ignore source off edge.  Could do better by looking at
            # source size.
            info.append(0)
            continue
        final = galsim.Convolve(obj.profile * exptime, psf)
        if chromatic:
            stamp = final.drawImage(
                bandpass, center=image_pos, wcs=imwcs.local(image_pos),
                method='phot', rng=rng)
        else:
            if obj.flux is not None:
                final = final.withFlux(
                    obj.flux[filter_name] * abflux)
                try:
                    stamp = final.drawImage(center=image_pos,
                                            wcs=imwcs.local(image_pos))
                except galsim.GalSimFFTSizeError:
                    log.warning(f'Skipping source {i} due to too '
                                f'large FFT needed for desired accuracy.')
        bounds = stamp.bounds & full_image.bounds
        if bounds.area() == 0:
            continue
        full_image[bounds] += stamp[bounds]
        nrender += 1
        info.append(time.time() - t0)
    log.info('Rendered %d sources...' % nrender)

    poisson_noise = galsim.PoissonNoise(rng)
    sky_image.addNoise(poisson_noise)

    # add Poisson noise if we made a noiseless, not-photon-shooting
    # image.
    if final is not None and not final.spectral:
        full_image.addNoise(poisson_noise)

    full_image += sky_image

    if darkrate is None:
        darkrate = roman.dark_current
    dark_image += darkrate * exptime
    dark_image.addNoise(poisson_noise)

    full_image.quantize()

    if return_info:
        full_image = (full_image, info)

    return full_image


def simulate(metadata, objlist,
             usecrds=True, webbpsf=True, level=2,
             seed=None, rng=None,
             **kwargs):
    """Simulate a sequence of observations on a field in different bandpasses.

    Parameters
    ----------
    metadata : dict
        metadata structure for Roman asdf file, including information about

        * pointing: metadata['pointing']['ra_v1'], metadata['pointing']['dec_v1']
        * date: metadata['exposure']['start_time']
        * sca: metadata['instrument']['detector']
        * bandpass: metadata['instrument']['optical_detector']
        * ma_table_number: metadata['exposure']['ma_table_number']

    objlist : list[CatalogObject]
        List of objects in the field to simulate
    usecrds : bool
        use CRDS to get distortion maps
    webbpsf : bool
        use webbpsf to generate PSF
    level : int
        0, 1 or 2, specifying level 1 or level 2 image
        0 makes a special idealized 'counts' image
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.

    Returns
    -------
    asdf structure with simulated image
    """

    all_metadata = copy.deepcopy(parameters.default_parameters_dictionary)
    flatmetadata = util.flatten_dictionary(metadata)
    flatmetadata = {'roman.meta'+k if k.find('roman.meta') != 0 else k: v
                    for k, v in flatmetadata.items()}
    all_metadata.update(**util.flatten_dictionary(metadata))
    ma_table_number = all_metadata['roman.meta.exposure.ma_table_number']
    sca = int(all_metadata['roman.meta.instrument.detector'][3:])
    coord = coordinates.SkyCoord(
        ra=all_metadata['roman.meta.pointing.ra_v1']*u.deg,
        dec=all_metadata['roman.meta.pointing.dec_v1']*u.deg)
    start_time = all_metadata['roman.meta.exposure.start_time']
    if not isinstance(start_time, astropy.time.Time):
        start_time = astropy.time.Time(start_time, format='isot')
    date = start_time
    filter_name = all_metadata['roman.meta.instrument.optical_element']

    ma_table = parameters.ma_table[ma_table_number]
    last_read = ma_table[-1][0] + ma_table[-1][1] - 1
    exptime = last_read * parameters.read_time

    if usecrds:
        reffiles = crds.getreferences(all_metadata, reftypes=['readnoise', 'dark'],
                                      observatory='roman')
        read_noise = asdf.open(reffiles['readnoise'])['roman']['data']
        darkrate = asdf.open(reffiles['dark'])['roman']['data']
        darkrate = darkrate[-1] / (np.mean(ma_table[-1])*parameters.read_time)
        nborder = parameters.nborder
        read_noise = read_noise[nborder:-nborder, nborder:-nborder]
        darkrate = darkrate[nborder:-nborder, nborder:-nborder]
    else:
        read_noise = galsim.roman.read_noise
        darkrate = galsim.roman.dark_current

    out = dict()

    if rng is None and seed is None:
        seed = 43
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)


    log.info('Simulating filter {0}...'.format(filter_name))
    counts = simulate_counts(
        sca, coord, date, objlist, filter_name, rng=rng,
        exptime=exptime, usecrds=usecrds, darkrate=darkrate,
        webbpsf=webbpsf)
    if level == 0:
        return counts
    l1 = romanisim.l1.make_l1(
        counts, ma_table_number, read_noise=read_noise, rng=rng, **kwargs)
    if level == 1:
        im = romanisim.l1.make_asdf(l1, metadata=all_metadata)
    elif level == 2:
        slopeinfo = make_l2(l1, ma_table, read_noise=read_noise)
        im = make_asdf(*slopeinfo, metadata=all_metadata)
    return im


def make_test_catalog_and_images(seed=12345, sca=7, filters=None, nobj=1000,
                                 return_variance=False, usecrds=True,
                                 webbpsf=True, **kwargs):
    """This routine kicks the tires on everything in this module."""
    log.info('Making catalog...')
    if filters is None:
        filters = ['Y106', 'J129', 'H158']
    metadata = copy.deepcopy(parameters.default_parameters_dictionary)
    coord = coordinates.SkyCoord(
        ra=metadata['roman.meta.pointing.ra_v1']*u.deg,
        dec=metadata['roman.meta.pointing.dec_v1']*u.deg)
    date = astropy.time.Time(
        metadata['roman.meta.exposure.start_time'],
        format='isot')
    imwcs = wcs.get_wcs(world_pos=coord, date=date, sca=sca, usecrds=usecrds)
    rd_sca = imwcs.toWorld(galsim.PositionD(
        roman.n_pix / 2 + 0.5, roman.n_pix / 2 + 0.5))
    cat = catalog.make_dummy_catalog(rd_sca, seed=seed, nobj=nobj)
    rng = galsim.UniformDeviate(0)
    out = dict()
    for filter_name in filters:
        im = simulate(metadata, objlist=cat, rng=rng, usecrds=usecrds, webbpsf=webbpsf,
                      **kwargs)
        out[filter_name] = im
    return out


def make_asdf(slope, slopevar_rn, slopevar_poisson, metadata=None,
              filepath=None):
    """Wrap a galsim simulated image with ASDF/roman_datamodel metadata.

    Eventually this needs to get enough info to reconstruct a refit WCS.
    """

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
    if metadata is not None:
        # ugly mess of flattening & unflattening to make sure that deeply
        # nested keywords all get propagated into the metadata structure.
        tmpmeta = util.flatten_dictionary(out['meta'])
        tmpmeta.update(util.flatten_dictionary(
            util.unflatten_dictionary(metadata)['roman']['meta']))
        out['meta'].update(util.unflatten_dictionary(tmpmeta))

    out['data'] = slope
    out['dq'] = (slope*0).astype('u4')
    out['var_poisson'] = slopevar_poisson
    out['var_rnoise'] = slopevar_rn
    out['var_flat'] = slope*0
    out['err'] = slope*0
    if filepath:
        af = asdf.AsdfFile()
        af.tree = {'roman': out}
        af.write_to(filepath)
    return out
