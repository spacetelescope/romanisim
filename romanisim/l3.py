"""Roman WFI simulator functions for Level 3 mosaics.

Based on galsim's implementation of Roman image simulation.  Uses galsim Roman modules
for most of the real work.
"""

import numpy as np
import math
import astropy.time
from astropy import table
import galsim
from galsim import roman

from . import parameters
from . import util
import romanisim.wcs
import romanisim.l1
import romanisim.bandpass
import romanisim.psf
import romanisim.image
import romanisim.persistence
from romanisim import log
import roman_datamodels.maker_utils as maker_utils

# Define centermost SCA for PSFs
CENTER_SCA = 2


def add_objects_to_l3(l3_mos, source_cat, rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_mos : MosaicModel
        Mosaic of images
    source_cat : list
        List of catalog objects to add to l3_mos

    Returns
    -------
    None
        l3_mos is updated in place
    """

    if rng is None and seed is None:
        seed = 143
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    # Obtain optical element
    filter_name = l3_mos.meta.basic.optical_element

    # Generate WCS
    twcs = romanisim.wcs.get_mosaic_wcs(l3_mos.meta, shape=l3_mos.data.shape)

    # Create PSF
    l3_psf = romanisim.psf.make_psf(filter_name=filter_name, sca=CENTER_SCA, chromatic=False, webbpsf=True)

    # Generate x,y positions for sources
    coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                       for o in source_cat])
    sourcecountsall = galsim.ImageF(l3_mos.data.shape[0], l3_mos.data.shape[1], wcs=twcs, xmin=0, ymin=0)
    # xpos, ypos = sourcecountsall.wcs._xy(coords[:, 0], coords[:, 1])
    xpos, ypos = sourcecountsall.wcs.radecToxy(coords[:, 0], coords[:, 1], 'rad')
    xpos_idx = [round(x) for x in xpos]
    ypos_idx = [round(y) for y in ypos]

    if ((min(xpos_idx) < 0) or (min(ypos_idx)) or (max(xpos_idx) > l3_mos.data.shape[0])
        or (max(ypos_idx) > l3_mos.data.shape[1])):
        log.error(f"A source is out of bounds! "
                  f"Source min x,y = {min(xpos_idx)},{min(ypos_idx)}, max = {max(xpos_idx)},{max(ypos_idx)} "
                  f"Image min x,y = {0},{0}, max = {l3_mos.data.shape[0]},{l3_mos.data.shape[1]}")

    # Create overall scaling factor map
    # Ct_all = (l3_mos.data.value / l3_mos.var_poisson)
    Ct_all = np.divide(l3_mos.data.value, l3_mos.var_poisson.value,
                       out=np.ones(l3_mos.data.shape), where=l3_mos.var_poisson.value != 0)

    # Cycle over sources and add them to the mosaic
    for idx, (x, y) in enumerate(zip(xpos_idx, ypos_idx)):
        # Set scaling factor for injected sources
        # Flux / sigma_p^2
        if l3_mos.var_poisson[x][y].value != 0:
            Ct = math.fabs(l3_mos.data[x][y].value / l3_mos.var_poisson[x][y].value)
        # elif l3_mos.data[x][y].value != 0:
        #     Ct = math.fabs(l3_mos.data[x][y].value)
        else:
            continue

        # Create empty postage stamp galsim source image
        sourcecounts = galsim.ImageF(l3_mos.data.shape[0], l3_mos.data.shape[1], wcs=twcs, xmin=0, ymin=0)

        # Simulate source postage stamp
        romanisim.image.add_objects_to_image(sourcecounts, [source_cat[idx]], xpos=[xpos[idx]],
                                             ypos=[ypos[idx]], psf=l3_psf, flux_to_counts_factor=Ct,
                                             bandpass=[filter_name], filter_name=filter_name, rng=rng)

        # Scale the source image back by its flux ratios
        sourcecounts /= Ct

        # Add sources to the original mosaic data array
        # l3_mos.data = (l3_mos.data.value + sourcecounts.array) * l3_mos.data.unit
        l3_mos.data = (l3_mos.data.value + np.swapaxes(sourcecounts.array, 0, 1)) * l3_mos.data.unit

        # Note for the future - other noise sources (read and flat) need to be implemented

    # Set new poisson variance
    # l3_mos.var_poisson = l3_mos.data.value / Ct_all
    l3_mos.var_poisson = np.divide(l3_mos.data.value, Ct_all, out=l3_mos.var_poisson.value,
                                   where=(Ct_all != 0)) * l3_mos.var_poisson.unit

    # l3_mos is updated in place, so no return
    return None


def make_l3(image, context, metadata, exptimes, rng=None, seed=None,
            saturation=None, unit_factor=1):
    """TBD
    """

    log.info('Apportioning counts to mosaic...')
    # resultants, dq = apportion_counts_to_resultants(
    #     counts.array, tij, inv_linearity=inv_linearity, crparam=crparam,
    #     persistence=persistence, tstart=tstart,
    #     rng=rng, seed=seed)

    # roman.addReciprocityFailure(resultants_object)

    # code from apportion_counts_to_resultants

    mosaic = np.swapaxes(image.array, 0, 1)

    # Set rng for creating readnoise, flat noise
    if rng is None and seed is None:
        seed = 46
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.GaussianDeviate(seed)

    # Need to add read noise, poisson noise, and flat noise

    # Simulate read and flat as random noise
    # if read_noise is not None and not isinstance(read_noise, u.Quantity):
    #     read_noise = read_noise * u.DN
    #     log.warning('Making up units for read noise.')
    # resultants are now in counts.
    # read noise is in counts.
    # log.info('Adding read noise...')
    # resultants = add_read_noise_to_resultants(
    #     resultants, tij, rng=rng, seed=seed,
    #     read_noise=read_noise)

    # Improve this for proper 3D contexts
    # ctx_2d = np.reshape(context[-2:], context.shape[-2:])
    # mosaic = np.divide(counts_arr, context[-2:], where=context[-2:]!=0, out=np.zeros(counts_arr.shape, dtype='f8'))
    # mosaic = np.divide(counts_arr, ctx_2d, where=ctx_2d!=0, out=np.zeros(counts_arr.shape, dtype='f8'))

    # Mosaics may suffer saturation? If so rework the following.
    # if saturation is None:
    #   saturation = parameters.reference_data['saturation']
    # mosaic = np.clip(mosaic, 0, 2 * 10**9).astype('i4')
    # this maybe should be better applied at read time?
    # it's not actually clear to me what the right thing to do
    # is in detail.
    # mosaic = np.clip(mosaic, 0 * u.DN, saturation)
    # mosaic = np.clip(mosaic, 0, saturation.value)
    # m = mosaic >= saturation.value
    # dq = np.zeros(counts_arr.shape, dtype=np.uint32)
    # dq[m] |= parameters.dqbits['saturated']

    # Set mosaic to be a mosaic model
    mosaic_mdl = maker_utils.mk_level3_mosaic(shape=mosaic.shape, meta=metadata)

    # Extract exposure time for each pixel
    exptimes_pix = util.decode_context_times(context, exptimes)

    # Set data
    mosaic_mdl.data = mosaic * unit_factor
    mosaic_mdl.data = np.divide(mosaic_mdl.data.value, exptimes_pix,
                                out=np.zeros(mosaic_mdl.data.shape),
                                where=exptimes_pix != 0) * mosaic_mdl.data.unit

    # Context
    # Binary index of images that contribute to the pixel
    # Defined by geometry.. if not, all non zero = 1.
    mosaic_mdl.context = context

    # Poisson noise
    mosaic_mdl.var_poisson = unit_factor**2 / exptimes_pix**2

    # Weight
    # Use exptime weight
    # ---
    # DQ used if set above for saturation?
    #  bitvalue = interpret_bit_flags(
    #           bitvalue, flag_name_map={dq.name: dq.value for dq in pixel} )
    #  if bitvalue is None:
    #      return np.ones(dqarr.shape, dtype=np.uint8)
    #  return np.logical_not(np.bitwise_and(dqarr, ~bitvalue)).astype(np.uint8)
    # ---
    #  dqmask = build_mask(model.dq, good_bits)
    #  elif weight_type == "exptime":
    #    exptime = model.meta.exposure.exposure_time
    #    result = exptime * dqmask

    # Fill everything else:
    # err, weight, var_rnoise, var_flat

    return mosaic_mdl


def generate_mosaic_geometry():
    """Generate a geometry map (context) for a mosaic dither pattern / set of pointings and roll angles
    TBD
    """
    pass


def simulate(metadata, cat, exptimes, context=None,
             usecrds=True, webbpsf=True, seed=None, rng=None,
             psf_keywords=dict(), **kwargs
             ):
    """TBD
    """

    if not usecrds:
        log.warning('--usecrds is not set.  romanisim will not use reference '
                    'files from CRDS.  The WCS may be incorrect and up-to-date '
                    'calibration information will not be used.')

    # Create metadata object
    meta = maker_utils.mk_mosaic_meta()
    meta['wcs'] = None

    for key in parameters.default_mosaic_parameters_dictionary.keys():
        meta[key].update(parameters.default_mosaic_parameters_dictionary[key])

    for key in metadata.keys():
        if key not in meta:
            meta[key] = metadata[key]
        # elif isinstance(meta[key], dict):
        else:
            meta[key].update(metadata[key])

    # May need an equivalent of this this for L3
    # util.add_more_metadata(meta)

    filter_name = metadata['basic']['optical_element']

    if rng is None and seed is None:
        seed = 43
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    if context is None:
        # Create geometry from the object list
        twcs = romanisim.wcs.get_mosaic_wcs(meta)

        coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                           for o in cat])

        allx, ally = twcs.radecToxy(coords[:, 0], coords[:, 1], 'rad')

        # Obtain the sample extremums
        xmin = min(allx)
        xmax = max(allx)
        ymin = min(ally)
        ymax = max(ally)

        # Obtain WCS center
        xcen, ycen = twcs.radecToxy(twcs.center.ra, twcs.center.dec, 'rad')

        # Determine maximum extremums from WCS center
        xdiff = max([math.ceil(xmax - xcen), math.ceil(xcen - xmin)])
        ydiff = max([math.ceil(ymax - ycen), math.ceil(ycen - ymin)])

        # Create context map preserving WCS center
        context = np.ones((1, 2 * xdiff, 2 * ydiff), dtype=np.uint32)

    log.info('Simulating filter {0}...'.format(filter_name))
    mosaic, simcatobj = simulate_cps(
        meta, cat, context, exptimes, rng=rng, usecrds=usecrds,
        webbpsf=webbpsf, psf_keywords=psf_keywords)

    extras = {}

    # if reffiles:
    #     extras["simulate_reffiles"] = {}
    #     for key, value in reffiles.items():
    #         extras["simulate_reffiles"][key] = value

    extras['simcatobj'] = simcatobj

    # TODO: Double check that this is a gwcs object
    # extras['wcs'] = wcs.convert_wcs_to_gwcs(mosaic.wcs)
    # extras['wcs'] = mosaic.wcs

    log.info('Simulation complete.')
    # return mosaic, extras
    return mosaic, extras


def simulate_cps(metadata, objlist, context, exptimes,
                 rng=None, seed=None, webbpsf=True, usecrds=False,
                 psf_keywords=dict()):
    """TBD
    """

    if rng is None and seed is None:
        seed = 43
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    filter_name = metadata['basic']['optical_element']

    date = metadata['basic']['time_mean_mjd']

    if not isinstance(date, astropy.time.Time):
        date = astropy.time.Time(date, format='mjd')

    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter_name]
    bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]

    # Generate WCS
    moswcs = romanisim.wcs.get_mosaic_wcs(metadata, shape=context.shape[-2:])

    # Is this needed?
    chromatic = False
    if (len(objlist) > 0
            and not isinstance(objlist, table.Table)  # this case is always gray
            and objlist[0].profile.spectral):
        chromatic = True

    # With context, is possible that individual PSFs could be properly assigned?
    # Would be a pain to implement
    psf = romanisim.psf.make_psf(filter_name=filter_name, wcs=moswcs, sca=CENTER_SCA,
                                 chromatic=chromatic, webbpsf=webbpsf,
                                 variable=True, **psf_keywords)

    # Create initial galsim image
    image = galsim.ImageF(context.shape[-2], context.shape[-1], wcs=moswcs, xmin=0, ymin=0)

    # Examine this in more detail to ensure it is correct
    # Create base sky level
    mos_cent_pos = moswcs.toWorld(image.true_center)
    sky_level = roman.getSkyLevel(bandpass, world_pos=mos_cent_pos,
                                  date=date.datetime, exptime=1)
    sky_level *= (1.0 + roman.stray_light_fraction)
    # sky_mosaic = image * 0
    # moswcs.makeSkyImage(sky_mosaic, sky_level)
    # sky_mosaic += roman.thermal_backgrounds[galsim_filter_name]
    abflux = romanisim.bandpass.get_abflux(filter_name)

    # Obtain physical unit conversion factor
    unit_factor = parameters.reference_data['photom'][filter_name]

    # Maybe return a mosaic object as well?
    mosaic, simcatobj = simulate_cps_generic(
        image, metadata, context, exptimes, objlist=objlist, psf=psf, zpflux=abflux, sky=sky_level,
        wcs=moswcs, rng=rng, seed=seed, unit_factor=unit_factor)

    return mosaic, simcatobj


def simulate_cps_generic(image, metadata, context, exptimes, objlist=None, psf=None,
                         zpflux=None, wcs=None, xpos=None, ypos=None, sky=None,
                         flat=None, rng=None, seed=None, unit_factor=1):
    """TBD
    """
    if rng is None and seed is None:
        seed = 144
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    # Objlist wcs check
    if (objlist is not None and len(objlist) > 0
            and image.wcs is None and (xpos is None or ypos is None)):
        raise ValueError('xpos and ypos must be set if rendering objects '
                         'without a WCS.')
    if objlist is None:
        objlist = []
    if len(objlist) > 0 and xpos is None:
        if isinstance(objlist, table.Table):
            xpos, ypos = image.wcs.radecToxy(
                np.radians(objlist['ra']), np.radians(objlist['dec']), 'rad')
        else:
            coord = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                             for o in objlist])
            xpos, ypos = image.wcs.radecToxy(coord[:, 0], coord[:, 1], 'rad')
        # use private vectorized transformation
    if xpos is not None:
        xpos = np.array(xpos)
    if ypos is not None:
        ypos = np.array(ypos)

    if len(objlist) > 0 and psf is None:
        raise ValueError('Must provide a PSF if you want to render objects.')

    if flat is None:
        flat = 1.0

    # # for some reason, galsim doesn't like multiplying an SED by 1, but it's
    # # okay with multiplying an SED by 1.0, so we cast to float.
    maxflat = float(np.max(flat))
    if maxflat > 1.1:
        log.warning('max(flat) > 1.1; this seems weird?!')
    if maxflat > 2:
        log.error('max(flat) > 2; this seems really weird?!')
    # how to deal with the flat field?  We artificially inflate the
    # exposure time of each source by maxflat when rendering.  And then we
    # do a binomial sampling of the total number of photons obtained per pixel
    # to figure out how many "should" have entered the pixel.

    chromatic = False
    if len(objlist) > 0 and objlist[0].profile.spectral:
        chromatic = True

    # add Poisson noise if we made a noiseless, not-photon-shooting
    # image.
    poisson_noise = galsim.PoissonNoise(rng, sky_level=sky)
    if not chromatic:
        image.addNoise(poisson_noise)

    # Add Objects requires a mosaic image
    # Maybe violates the notion of a "generic" count simulator?
    mosaic_mdl = make_l3(image, context, metadata, exptimes, unit_factor=unit_factor)

    if wcs is not None:
        mosaic_mdl['wcs'] = wcs

    # Add objects to mosaic model
    objinfo = add_objects_to_l3(
        mosaic_mdl, objlist, rng=rng)
    objinfo = np.zeros(
        len(objlist),
        dtype=[('x', 'f4'), ('y', 'f4'), ('counts', 'f4'), ('time', 'f4')])
    if len(objlist) > 0:
        objinfo['x'] = xpos
        objinfo['y'] = ypos

    return mosaic_mdl, objinfo
