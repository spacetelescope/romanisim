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
import roman_datamodels.datamodels as rdm
from roman_datamodels.stnode import WfiMosaic
import astropy.units as u


def add_objects_to_l3(l3_mos, source_cat, exptimes, xpos=None, ypos=None, coords=None, cps_conv=1.0, unit_factor=1.0,
                      filter_name=None, coords_unit='rad', wcs=None, psf=None, rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_mos : MosaicModel or galsim.Image
        Mosaic of images
    source_cat : list
        List of catalog objects to add to l3_mos
    exptimes : list
        Exposure times to scale back to rate units
    xpos, ypos : array_like
        x & y positions of sources (pixel) at which sources should be added
    coords : array_like
        ra & dec positions of sources (coords_unit) at which sources should be added
    cps_conv : float
        Factor to convert data to cps
    unit_factor: float
        Factor to convert counts data to MJy / sr
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
    if filter_name is None:
        filter_name = l3_mos.meta.basic.optical_element

    # Generate WCS (if needed)
    if wcs is None:
        wcs = romanisim.wcs.get_mosaic_wcs(l3_mos.meta, shape=l3_mos.data.shape)

    # Create PSF (if needed)
    if psf is None:
        psf = romanisim.psf.make_psf(filter_name=filter_name, sca=parameters.default_sca, chromatic=False, webbpsf=True)

    # Create Image canvas to add objects to
    if isinstance(l3_mos, (rdm.MosaicModel, WfiMosaic)):
        sourcecountsall = galsim.ImageF(l3_mos.data.value, wcs=wcs, xmin=0, ymin=0)
    else:
        sourcecountsall = l3_mos

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
                                         psf=psf, flux_to_counts_factor=[xpt * cps_conv for xpt in exptimes],
                                         convtimes=[xpt / unit_factor for xpt in exptimes],
                                         bandpass=[filter_name], filter_name=filter_name,
                                         rng=rng, seed=seed)

    # Save array with added sources
    if isinstance(l3_mos, (rdm.MosaicModel, WfiMosaic)):
        l3_mos.data = sourcecountsall.array * l3_mos.data.unit

    # l3_mos is updated in place, so no return
    return None


def generate_mosaic_geometry():
    """Generate a geometry map (context) for a mosaic dither pattern / set of pointings and roll angles
    TBD
    """
    pass

def generate_exptime_array(cat, meta):
    """ Create geometry from the object list
    TBD
    """

    # Get wcs for this metadata
    twcs = romanisim.wcs.get_mosaic_wcs(meta)

    # Obtain sky positions for objects
    coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                        for o in cat])

    # Calculate x,y positions for objects
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
    context = np.ones((1, 2 * ydiff, 2 * xdiff), dtype=np.uint32)

    return context


def simulate(shape, wcs, efftimes, filter, catalog, metadata={}, effreadnoise=None, sky=None, psf=None, seed=None, rng=None):
    """TBD
    """

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

    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter]
    bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]

    # Create initial galsim image (x & y are flipped)
    image = galsim.ImageF(shape[1], shape[0], wcs=wcs, xmin=0, ymin=0)

    if sky is None:
        date = meta['basic']['time_mean_mjd']
        if not isinstance(date, astropy.time.Time):
            date = astropy.time.Time(date, format='mjd')

        # Examine this in more detail to ensure it is correct
        # Create base sky level
        mos_cent_pos = wcs.toWorld(image.true_center)
        # sky_level = roman.getSkyLevel(bandpass, world_pos=mos_cent_pos,
        #                             date=date.datetime, exptime=1)
        sky_level = roman.getSkyLevel(bandpass, world_pos=mos_cent_pos, exptime=1)
        sky_level *= (1.0 + roman.stray_light_fraction)
        # sky_mosaic = image * 0
        # wcs.makeSkyImage(sky_mosaic, sky_level)
        # sky_mosaic += roman.thermal_backgrounds[galsim_filter_name]
        sky = sky_level

    abflux = romanisim.bandpass.get_abflux(filter)

    # Obtain unit conversion factors
    # Need to convert from counts / pixel to MJy / sr
    # Flux to counts
    cps_conv = romanisim.bandpass.get_abflux(filter)
    # Unit factor
    unit_factor = ((3631 * u.Jy) / (romanisim.bandpass.get_abflux(filter) * 10e6
                                    * parameters.reference_data['photom']["pixelareasr"][filter])).to(u.MJy / u.sr)

    # Set effective read noise
    if effreadnoise is None:
        effreadnoise = efftimes / parameters.read_time

    # Simulate mosaic
    mosaic, simcatobj = simulate_cps(
        image, meta, efftimes, objlist=catalog, psf=psf, zpflux=abflux, sky=sky,
        effreadnoise=effreadnoise, cps_conv=cps_conv,
        wcs=wcs, rng=rng, seed=seed, unit_factor=unit_factor)

    # Create Mosaic Model
    mosaic_mdl = make_l3(mosaic, metadata, efftimes, unit_factor=unit_factor)

    # Add simulation artifacts
    extras = {}
    extras['simcatobj'] = simcatobj
    extras['wcs'] = mosaic.wcs

    log.info('Simulation complete.')
    # return mosaic, extras
    return mosaic_mdl, extras


def simulate_cps(image, metadata, efftimes, objlist=None, psf=None,
                         zpflux=None, wcs=None, xpos=None, ypos=None, sky=None,
                         effreadnoise=None, cps_conv=1,
                         flat=None, rng=None, seed=None, unit_factor=1,
                         ignore_distant_sources=10,):
    """TBD
    """
    if rng is None and seed is None:
        seed = 144
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    # Dictionary to hold simulation artifacts
    extras = {}

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

    # Check for objects outside the image boundary
    if len(objlist) > 0:
        keep = romanisim.image.in_bounds(xpos, ypos, image.bounds, ignore_distant_sources)
    else:
        keep = []

    chromatic = False
    if len(objlist) > 0 and objlist[0].profile.spectral:
        chromatic = True

    

    # Pixelized object locations
    xpos_idx = [round(x) for x in xpos]
    ypos_idx = [round(y) for y in ypos]

    # Set exposure time per source
    if isinstance(efftimes, np.ndarray):
        src_exptimes = [efftimes[y,x] for x,y in zip(xpos_idx, ypos_idx)]
    else:
        src_exptimes = [efftimes] * len(xpos)
    src_exptimes = np.array(src_exptimes)

    # Set the average exposure time to objects lacking one
    avg_exptime = np.average(src_exptimes)
    src_exptimes[keep] = avg_exptime



    # Noise

    # add Poisson noise if we made a noiseless, not-photon-shooting
    # image.
    poisson_noise = galsim.PoissonNoise(rng, sky_level=sky)
    if not chromatic:
        image.addNoise(poisson_noise)

    if effreadnoise is not None:
        readnoise = np.zeros(image.array.shape, dtype='f4')
        rn_rng = galsim.GaussianDeviate(seed)
        rn_rng.generate(readnoise)
        readnoise = readnoise * effreadnoise
        readnoise /= np.sqrt(efftimes)

        image += readnoise
        extras['readnoise'] = readnoise

    # Convert to the proper units and temporal scaling
    image *= unit_factor.value
    image /= efftimes

    # Generate GWCS compatible wcs
    sipwcs = romanisim.wcs.get_mosaic_wcs(metadata, shape=image.array.shape, xpos=xpos, ypos=ypos, coord=coord)
    image.wcs = sipwcs

    # Add objects to mosaic
    objinfo = add_objects_to_l3(
        image, objlist, src_exptimes, wcs=sipwcs, filter_name=metadata['basic']['optical_element'], rng=rng, cps_conv=cps_conv, unit_factor=unit_factor)
       
    # Add object info artifacts
    objinfo = np.zeros(
        len(objlist),
        dtype=[('x', 'f4'), ('y', 'f4'), ('counts', 'f4'), ('time', 'f4')])
    if len(objlist) > 0:
        objinfo['x'] = xpos
        objinfo['y'] = ypos
    extras['objinfo'] = objinfo

    return image, extras

def make_l3(image, metadata, efftimes, rng=None, seed=None,
            var_flat=None, var_readnoise=None, context=None,
            unit_factor=(1.0 * u.MJy / u.sr)):
    """TBD
    """

    mosaic = image.array.copy()

    # Set rng for creating readnoise, flat noise
    if rng is None and seed is None:
        seed = 46
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.GaussianDeviate(seed)

    # Ensure that effective times are an array
    if isinstance(efftimes, np.ndarray):
        efftimes_arr = efftimes
    else:
        efftimes_arr = efftimes * np.ones(mosaic.shape, dtype=np.float32)

    # Set mosaic to be a mosaic node
    mosaic_node = maker_utils.mk_level3_mosaic(shape=mosaic.shape, meta=metadata)

    # Set data
    mosaic_node.data = mosaic * unit_factor.unit

    # Poisson noise variance
    # mosaic_node.var_poisson = unit_factor**2 / efftimes**2
    mosaic_node.var_poisson = (unit_factor**2 / efftimes_arr**2).astype(np.float32)
   
    # Read noise variance
    if var_readnoise is None:
        mosaic_node.var_rnoise = u.Quantity(np.zeros(mosaic.shape, dtype=np.float32), unit_factor.unit ** 2)
    else:
        mosaic_node.var_rnoise = u.Quantity(var_readnoise, unit_factor.unit ** 2, dtype=np.float32)

    # Flat variance
    if var_flat is None:
        mosaic_node.var_flat = u.Quantity(np.zeros(mosaic.shape, dtype=np.float32), unit_factor.unit ** 2)
    else:
        mosaic_node.var_flat = var_flat

    # Total error
    mosaic_node.err = np.sqrt(mosaic_node.var_rnoise + mosaic_node.var_flat + mosaic_node.var_poisson).astype(np.float32)

    # Weight
    # Use exptime weight
    mosaic_node.weight = efftimes_arr.astype(np.float32)

    # Context
    # Binary index of images that contribute to the pixel
    # Defined by geometry.. if not, all non zero = 1.
    if context is None:
        mosaic_node.context = np.ones((1,) + mosaic.shape, dtype=np.uint32)
    else:
        mosaic_node.context = context

    # Set mosaic to be a mosaic model
    wfi_mosaic = rdm.MosaicModel(mosaic_node)

    return wfi_mosaic
