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


def add_objects_to_l3(l3_mos, source_cat, exptimes, xpos=None, ypos=None, coords=None, unit_factor=1.0,
                      coords_unit='rad', wcs=None, psf=None, rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_mos : MosaicModel
        Mosaic of images
    source_cat : list
        List of catalog objects to add to l3_mos
    exptimes : list
        Exposure times to scale back to rate units
    xpos, ypos : array_like
        x & y positions of sources (pixel) at which sources should be added
    coords : array_like
        ra & dec positions of sources (coords_unit) at which sources should be added
    unit_factor: float
        Factor to convert data to MJy / sr
    coords_unit : string
        units of coords
    wcs : galsim.GSFitsWCS
        WCS corresponding to image
    psf : galsim.Profile
        PSF for image
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    None
        l3_mos is updated in place
    """

    # Obtain optical element
    filter_name = l3_mos.meta.basic.optical_element

    # Generate WCS (if needed)
    if wcs is None:
        wcs = romanisim.wcs.get_mosaic_wcs(l3_mos.meta, shape=l3_mos.data.shape)

    # Create PSF (if needed)
    if psf is None:
        psf = romanisim.psf.make_psf(filter_name=filter_name, sca=CENTER_SCA, chromatic=False, webbpsf=True)

    # Create Image canvas to add objects to
    sourcecountsall = galsim.ImageF(l3_mos.data.value, wcs=wcs, xmin=0, ymin=0)

    # Create position arrays (if needed)
    if any(pos is None for pos in [xpos, ypos]):
        # Create coordinates (if needed)
        if coords is None:
            coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                               for o in source_cat])
            coords_unit = 'rad'
        # Generate x,y positions for sources
        xpos, ypos = sourcecountsall.wcs.radecToxy(coords[:, 0], coords[:, 1], coords_unit)

    # Add sources to the original mosaic data array
    romanisim.image.add_objects_to_image(sourcecountsall, source_cat, xpos=xpos, ypos=ypos,
                                         psf=psf, flux_to_counts_factor=[xpt * unit_factor for xpt in exptimes],
                                         exptimes=exptimes, bandpass=[filter_name], filter_name=filter_name,
                                         wcs=wcs, rng=rng, seed=seed)

    # Save array with added sources
    l3_mos.data = sourcecountsall.array * l3_mos.data.unit

    # l3_mos is updated in place, so no return
    return None


def generate_mosaic_geometry():
    """Generate a geometry map (context) for a mosaic dither pattern / set of pointings and roll angles
    TBD
    """
    pass

def generate_exptime_array():
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

    print(f"XXX 2*xdiff = {2*xdiff}")
    print(f"XXX 2*ydiff = {2*ydiff}")

    # Create context map preserving WCS center
    # context = np.ones((1, 2 * xdiff, 2 * ydiff), dtype=np.uint32)
    context = np.ones((1, 2 * ydiff, 2 * xdiff), dtype=np.uint32)


def simulate(shape, wcs, efftimes, filter, catalog, metadata={}, effreadnoise=None, sky=None, psf=None, seed=None, rng=None):
            # , cat, exptimes, context=None,
            #  usecrds=True, webbpsf=True, seed=None, rng=None,
            #  psf_keywords=dict(), **kwargs
            #  ):
    """TBD
    """

    # if not usecrds:
    #     log.warning('--usecrds is not set.  romanisim will not use reference '
    #                 'files from CRDS.  The WCS may be incorrect and up-to-date '
    #                 'calibration information will not be used.')

    # Create metadata object
    meta = maker_utils.mk_mosaic_meta()

    for key in parameters.default_mosaic_parameters_dictionary.keys():
        meta[key].update(parameters.default_mosaic_parameters_dictionary[key])

    meta['wcs'] = wcs
    meta['basic']['optical_element'] = filter

    for key in metadata.keys():
        if key not in meta:
            meta[key] = metadata[key]
        # elif isinstance(meta[key], dict):
        else:
            meta[key].update(metadata[key])

    # May need an equivalent of this this for L3?
    # util.add_more_metadata(meta)

    log.info('Simulating filter {0}...'.format(filter))
    # mosaic, simcatobj = simulate_cps(
    #     meta, cat, context, exptimes, rng=rng, usecrds=usecrds,
    #     webbpsf=webbpsf, psf_keywords=psf_keywords)
    # ORIG sim_CPS code below

    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter]
    bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]

    # objlist = catalog

    # # Is this needed?
    # chromatic = False
    # if (len(objlist) > 0
    #         and not isinstance(objlist, table.Table)  # this case is always gray
    #         and objlist[0].profile.spectral):
    #     chromatic = True

    # Create initial galsim image (x & y are flipped)
    image = galsim.ImageF(shape[0], shape[1], wcs=wcs, xmin=0, ymin=0)
    print(f"XXX image.array.shape = {image.array.shape}")

    if sky is None:
        date = meta['basic']['time_mean_mjd']
        if not isinstance(date, astropy.time.Time):
            date = astropy.time.Time(date, format='mjd')

        # Examine this in more detail to ensure it is correct
        # Create base sky level
        mos_cent_pos = wcs.toWorld(image.true_center)
        sky_level = roman.getSkyLevel(bandpass, world_pos=mos_cent_pos,
                                    date=date.datetime, exptime=1)
        sky_level *= (1.0 + roman.stray_light_fraction)
        # sky_mosaic = image * 0
        # wcs.makeSkyImage(sky_mosaic, sky_level)
        # sky_mosaic += roman.thermal_backgrounds[galsim_filter_name]
        sky = sky_level

    abflux = romanisim.bandpass.get_abflux(filter)

    # Obtain physical unit conversion factor
    unit_factor = parameters.reference_data['photom'][filter]

    # mosaic, simcatobj = simulate_cps(
    #     meta, cat, context, exptimes, rng=rng, usecrds=usecrds,
    #     webbpsf=webbpsf, psf_keywords=psf_keywords) 
    # renamed simulate_cps_generic
    # Maybe return a mosaic object as well?
    mosaic, simcatobj = simulate_cps(
        image, meta, efftimes, objlist=catalog, psf=psf, zpflux=abflux, sky=sky,
        wcs=wcs, rng=rng, seed=seed, unit_factor=unit_factor)

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


# def simulate_cps_generic(image, metadata, context, exptimes, objlist=None, psf=None,
#                          zpflux=None, wcs=None, xpos=None, ypos=None, sky=None,
#                          flat=None, rng=None, seed=None, unit_factor=1):
def simulate_cps(image, metadata, efftimes, objlist=None, psf=None,
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

    # if len(objlist) > 0 and psf is None:
    #     raise ValueError('Must provide a PSF if you want to render objects.')

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
    mosaic_mdl = make_l3(image, metadata, efftimes, unit_factor=unit_factor)

    if wcs is not None:
        mosaic_mdl['wcs'] = wcs

    # Extract exposure time for each pixel
    # exptimes_pix = util.decode_context_times(context, exptimes)

    # print(f"XXX efftimes.shape = {efftimes.shape}")

    xpos_idx = [round(x) for x in xpos]
    ypos_idx = [round(y) for y in ypos]

    print(f"XXX Center x,y = {image.wcs.radecToxy(image.wcs.center.ra.rad, image.wcs.center.dec.rad, 'rad')}")

    print(f"XXX max, min (xpos) = {max(xpos)},{min(xpos)}")
    print(f"XXX max, min (ypos) = {max(ypos)},{min(ypos)}")

    print(f"XXX max, min (xpos_idx) = {max(xpos_idx)},{min(xpos_idx)}")
    print(f"XXX max, min (ypos_idx) = {max(ypos_idx)},{min(ypos_idx)}")

    if isinstance(efftimes, np.ndarray):
        print(f"XXX efftimes is an array!")
        src_exptimes = [efftimes[y,x] for x,y in zip(xpos_idx, ypos_idx)]
        # src_exptimes = [efftimes[x,y] for x,y in zip(xpos_idx, ypos_idx)]
    else:
        src_exptimes = [efftimes] * len(xpos)

    # Add objects to mosaic model
    objinfo = add_objects_to_l3(
        mosaic_mdl, objlist, src_exptimes, rng=rng)

    objinfo = np.zeros(
        len(objlist),
        dtype=[('x', 'f4'), ('y', 'f4'), ('counts', 'f4'), ('time', 'f4')])
    if len(objlist) > 0:
        objinfo['x'] = xpos
        objinfo['y'] = ypos

    return mosaic_mdl, objinfo


# def make_l3(image, context, metadata, exptimes, rng=None, seed=None,
#             saturation=None, unit_factor=1):
def make_l3(image, metadata, efftimes, rng=None, seed=None,
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
    # mosaic = image.array.copy()

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

    # # Extract exposure time for each pixel
    # exptimes_pix = util.decode_context_times(context, exptimes)

    # Set data
    mosaic_mdl.data = mosaic * unit_factor
    print(f"XXX mosaic_mdl.data.value.shape = {mosaic_mdl.data.value.shape}")
    mosaic_mdl.data = np.divide(mosaic_mdl.data.value, efftimes,
                                out=np.zeros(mosaic_mdl.data.shape),
                                where=efftimes != 0) * mosaic_mdl.data.unit

    # Context
    # Binary index of images that contribute to the pixel
    # Defined by geometry.. if not, all non zero = 1.
    # mosaic_mdl.context = context

    # Poisson noise
    # mosaic_mdl.var_poisson = unit_factor**2 / efftimes**2

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
