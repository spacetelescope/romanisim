"""Roman WFI simulator functions for Level 3 mosaics.

Based on galsim's implementation of Roman image simulation.  Uses galsim Roman modules
for most of the real work.
"""

import math
import numpy as np
import galsim

from . import log
import romanisim.catalog
import romanisim.wcs
import romanisim.l1
import romanisim.bandpass
import romanisim.psf
import romanisim.image
import romanisim.persistence
import romanisim.parameters
import roman_datamodels.datamodels as rdm
from roman_datamodels.stnode import WfiMosaic
import astropy.units as u
import roman_datamodels.maker_utils as maker_utils
import astropy
from galsim import roman
from astropy import table


def add_objects_to_l3(l3_mos, source_cat, exptimes, xpos=None, ypos=None, coords=None, cps_conv=1.0, unit_factor=1.0,
                      filter_name=None, bandpass=None, coords_unit='rad', wcs=None, psf=None, rng=None, seed=None):
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
    filter_name : str
        Filter to use to select appropriate flux from objlist. This is only
        used when achromatic PSFs and sources are being rendered.
    bandpass : galsim.Bandpass
        Bandpass in which mosaic is being rendered. This is used only in cases
        where chromatic profiles & PSFs are being used.
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
    if bandpass is None:
        bandpass = [filter_name]

    # Generate WCS (if needed)
    if wcs is None:
        wcs = romanisim.wcs.get_mosaic_wcs(l3_mos.meta, shape=l3_mos.data.shape)

    # Create PSF (if needed)
    if psf is None:
        psf = romanisim.psf.make_psf(filter_name=filter_name, sca=romanisim.parameters.default_sca, chromatic=False, webbpsf=True)

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
    outinfo = romanisim.image.add_objects_to_image(sourcecountsall, source_cat, xpos=xpos, ypos=ypos,
                                         psf=psf, flux_to_counts_factor=[xpt * cps_conv for xpt in exptimes],
                                         convtimes=[xpt / unit_factor for xpt in exptimes],
                                         bandpass=bandpass, filter_name=filter_name,
                                         rng=rng, seed=seed)

    # Save array with added sources
    if isinstance(l3_mos, (rdm.MosaicModel, WfiMosaic)):
        l3_mos.data = sourcecountsall.array * l3_mos.data.unit

    # l3_mos is updated in place, so no return
    return outinfo


def generate_mosaic_geometry():
    """Generate a geometry map (context) for a mosaic dither pattern / set of pointings and roll angles
    TBD
    """
    pass


def generate_exptime_array(cat, meta):
    """ To be updated.
    Function to ascertain / set exposure time in each pixel
    Present code is placeholder to be built upon
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


def simulate(shape, wcs, efftimes, filter, catalog, metadata={},
             effreadnoise=None, sky=None, psf=None, coords_unit='rad',
             cps_conv=None, unit_factor=None,
             bandpass=None, seed=None, rng=None,
             **kwargs):
    """Simulate a sequence of observations on a field in different bandpasses.

    Parameters
    ----------
    shape : tuple
        Array shape of mosaic to simulate.
    wcs : galsim.GSFitsWCS
        WCS corresponding to image
    efftimes : np.ndarray or float
        Effective exposure time of reach pixel in mosaic.
        If an array, shape must match shape parameter.
    filter : str
        Filter to use to select appropriate flux from objlist. This is only
        used when achromatic PSFs and sources are being rendered.
    catalog : list[CatalogObject] or Table
        List of catalog objects to add to l3_mos
    metadata : dict
        Metadata structure for Roman asdf file.
    effreadnoise : float
        Effective read noise for mosaic.
    sky : float or array_like
        Image or constant with the counts / pix / sec from sky. If None, then
        sky will be generated from galsim's getSkyLevel for Roman for the
        date provided in metadata[basic][time_mean_mjd].
    psf : galsim.Profile
        PSF for image
    coords_unit : string
        units of coords
    cps_conv : float
        Factor to convert data to cps
    unit_factor: float
        Factor to convert counts data to MJy / sr
    bandpass : galsim.Bandpass
        Bandpass in which mosaic is being rendered. This is used only in cases
        where chromatic profiles & PSFs are being used.
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    mosaic_mdl : roman_datamodels model
        simulated mosaic
    extras : dict
        Dictionary of additionally valuable quantities.  Includes at least
        simcatobj, the image positions and fluxes of simulated objects.  It may
        also include information on persistence and cosmic ray hits.
    """

    # Create metadata object
    meta = maker_utils.mk_mosaic_meta()
    for key in romanisim.parameters.default_mosaic_parameters_dictionary.keys():
        meta[key].update(romanisim.parameters.default_mosaic_parameters_dictionary[key])
    meta['wcs'] = wcs
    meta['basic']['optical_element'] = filter
    for key in metadata.keys():
        if key not in meta:
            meta[key] = metadata[key]
        else:
            meta[key].update(metadata[key])
    # May need an equivalent of this this for L3?
    # util.add_more_metadata(meta)

    log.info('Simulating filter {0}...'.format(filter))

    # Get filter and bandpass
    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter]
    if bandpass is None:
        bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]

    # Create initial galsim image (x & y are flipped)
    image = galsim.ImageF(shape[1], shape[0], wcs=wcs, xmin=0, ymin=0)

    # Create sky for this mosaic, if not provided (in cps)
    if sky is None:
        date = meta['basic']['time_mean_mjd']
        if not isinstance(date, astropy.time.Time):
            date = astropy.time.Time(date, format='mjd')

        # Examine this in more detail to ensure it is correct
        # Create base sky level
        mos_cent_pos = wcs.toWorld(image.true_center)
        sky_level = roman.getSkyLevel(bandpass, world_pos=mos_cent_pos, exptime=1)
        sky_level *= (1.0 + roman.stray_light_fraction)
        sky_mosaic = image * 0
        wcs.makeSkyImage(sky_mosaic, sky_level)
        sky_mosaic += roman.thermal_backgrounds[galsim_filter_name]
        sky = sky_mosaic

    # Obtain unit conversion factors
    # Need to convert from counts / pixel to MJy / sr
    # Flux to counts
    if cps_conv is None:
        cps_conv = romanisim.bandpass.get_abflux(filter)
    # Unit factor
    if unit_factor is None:
        unit_factor = ((3631 * u.Jy) / (romanisim.bandpass.get_abflux(filter) * 10e6
                        * romanisim.parameters.reference_data['photom']["pixelareasr"][filter])).to(u.MJy / u.sr)

    # Set effective read noise
    if effreadnoise is None:
        effreadnoise = efftimes / romanisim.parameters.read_time

    # Simulate mosaic cps
    mosaic, simcatobj = simulate_cps(
        image, meta, efftimes, objlist=catalog, psf=psf, 
        sky=sky,
        effreadnoise=effreadnoise, cps_conv=cps_conv, coords_unit=coords_unit,
        bandpass=bandpass, 
        wcs=wcs, rng=rng, seed=seed, unit_factor=unit_factor,
        **kwargs)

    # Set poisson variance
    if "var_poisson" in simcatobj:
        var_poisson = simcatobj["var_poisson"]
    else:
        var_poisson = None

    # Set read noise variance
    if "var_readnoise" in simcatobj:
        var_readnoise = simcatobj["var_readnoise"].value
    else:
        var_readnoise = None

    # Create Mosaic Model
    mosaic_mdl = make_l3(mosaic, metadata, efftimes, unit_factor=unit_factor,
                         var_poisson=var_poisson, var_readnoise=var_readnoise)

    # Add simulation artifacts
    extras = {}
    extras['simcatobj'] = simcatobj
    extras['wcs'] = mosaic.wcs

    log.info('Simulation complete.')
    # return mosaic, extras
    return mosaic_mdl, extras


def simulate_cps(image, metadata, efftimes, objlist=None, psf=None,
                         wcs=None, xpos=None, ypos=None, coord=None, sky=0,
                         effreadnoise=None, cps_conv=1, unit_factor=1.0 * (u.MJy / u.sr),
                         bandpass=None, coords_unit='rad',
                         rng=None, seed=None, ignore_distant_sources=10,):
    """Simulate average MegaJankies per steradian in a single SCA.

    Parameters
    ----------
    image : galsim.Image
        Image onto which other effects should be added, with associated WCS.
    metadata : dict
        Metadata structure for Roman asdf file.
    efftimes : np.ndarray or float
        Effective exposure time of reach pixel in mosaic.
        If an array, shape must match shape parameter.
    objlist : list[CatalogObject], Table, or None
        Sources to render
    psf : galsim.Profile
        PSF to use when rendering sources
    wcs : galsim.GSFitsWCS
        WCS corresponding to image
    xpos, ypos : array_like (float)
        x, y positions of each source in objlist
    coord : array_like (float)
        ra, dec positions of each source in objlist
    sky : float or array_like
        Image or constant with the counts / pix / sec from sky.
    effreadnoise : float
        Effective read noise for mosaic.
    cps_conv : float
        Factor to convert data to cps
    unit_factor: float
        Factor to convert counts data to MJy / sr
    bandpass : galsim.Bandpass
        bandpass to use for rendering chromatic objects
    coords_unit : string
        units of coords
    rng : galsim.BaseDeviate
        random number generator
    seed : int
        seed for random number generator
    ignore_distant_sources : int
        Ignore sources more than this distance off image.

    Returns
    -------
    objinfo : np.ndarray
        Information on position and flux of each rendered source.

    Returns
    -------
    image : galsim.Image
        Idealized image of scene as seen by Roman
    extras : dict
        catalog of simulated objects in image, noise, and misc. debug
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
            coord = np.array([[o['ra'], np.radians(o['dec'])] for o in objlist])
        else:
            if (coords_unit == 'rad'):
                coord = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                                for o in objlist])
            else:
                coord = np.array([[o.sky_pos.ra.deg, o.sky_pos.dec.deg]
                                for o in objlist])
        xpos, ypos = image.wcs.radecToxy(coord[:, 0], coord[:, 1], 'rad')
        # use private vectorized transformation
    if xpos is not None:
        xpos = np.array(xpos)
    if ypos is not None:
        ypos = np.array(ypos)

    chromatic = False
    if len(objlist) > 0 and objlist[0].profile.spectral:
        chromatic = True

    # Check for objects outside the image boundary (+ consideration)
    if len(objlist) > 0:
        keep = romanisim.image.in_bounds(xpos, ypos, image.bounds, ignore_distant_sources)

        objlist = np.array(objlist)[keep].tolist()
        xpos = xpos[keep]
        ypos = ypos[keep]
        # Pixelized object locations
        xpos_idx = [round(x) for x in xpos]
        ypos_idx = [round(y) for y in ypos]

        offedge = romanisim.image.in_bounds(xpos, ypos, image.bounds, 0)
        # Set exposure time per source
        if isinstance(efftimes, np.ndarray):
            src_exptimes = [efftimes[y, x] for x, y in zip(xpos_idx, ypos_idx)]
        else:
            src_exptimes = [efftimes] * len(xpos)
        src_exptimes = np.array(src_exptimes)

        # Set the average exposure time to objects lacking one
        if True in offedge:
            avg_exptime = np.average(src_exptimes)
            src_exptimes[offedge] = avg_exptime

    else:
        src_exptimes = []

    # Generate GWCS compatible wcs
    if wcs is None:
        wcs = romanisim.wcs.get_mosaic_wcs(metadata, shape=image.array.shape, xpos=xpos, ypos=ypos, coord=coord)
    image.wcs = wcs

    # Add objects to mosaic
    if len(src_exptimes) > 0:
        objinfokeep = add_objects_to_l3(
            image, objlist, src_exptimes, wcs=wcs, xpos=xpos, ypos=ypos,
            filter_name=metadata['basic']['optical_element'], bandpass=bandpass, psf=psf, rng=rng,
            cps_conv=cps_conv, unit_factor=unit_factor)

    # Add object info artifacts
    objinfo = {}
    objinfo['array'] = np.zeros(
        len(objlist),
        dtype=[('x', 'f4'), ('y', 'f4'), ('counts', 'f4'), ('time', 'f4')])
    if len(objlist) > 0:
        objinfo['x'] = xpos
        objinfo['y'] = ypos
        objinfo['counts'] = objinfokeep['counts']
        objinfo['time'] = objinfokeep['time']
    else:
        objinfo['counts'] = np.array([])
        objinfo['time'] = np.array([])
    extras['objinfo'] = objinfo

    # Noise
    im_no_noise = image.array.copy()

    # add Poisson noise if we made a noiseless, not-photon-shooting
    # image.
    poisson_noise = galsim.PoissonNoise(rng)
    if not chromatic:
        # This works in ADU, so need to convert back to counts first.. add this, then convert back
        image *= efftimes / unit_factor.value
        image.addNoise(poisson_noise)
        image /= efftimes / unit_factor.value

    if sky is not None:
        if isinstance(sky, (galsim.Image)):
            sky_arr = sky.array
        elif not isinstance(sky, (np.ndarray)):
            sky_arr = sky * np.ones(image.array.shape, dtype=float)
        else:
            sky_arr = sky

        workim = image * 0
        workim += sky
        workim *= efftimes
        workim.addNoise(poisson_noise)

        image += (workim * unit_factor.value / efftimes)

    else:
        sky_arr = np.zeros(image.array.shape, dtype=float)

    extras['var_poisson'] = np.abs(image.array - (im_no_noise + sky_arr))

    # Add readnoise
    if effreadnoise is not None:
        # This works in ADU, so need to convert back to counts first.. add this, then convert back
        readnoise = np.zeros(image.array.shape, dtype='f4')
        rn_rng = galsim.GaussianDeviate(seed)
        rn_rng.generate(readnoise)
        readnoise = readnoise * effreadnoise
        readnoise /= np.sqrt(efftimes)

        image += readnoise * unit_factor.value / efftimes
        extras['readnoise'] = readnoise * unit_factor / efftimes

        extras['var_readnoise'] = (rn_rng.sigma * (effreadnoise / np.sqrt(efftimes)) * (unit_factor / efftimes))**2

    # Return image and artifacts
    return image, extras


def make_l3(image, metadata, efftimes, rng=None, seed=None, var_poisson=None,
            var_flat=None, var_readnoise=None, context=None,
            unit_factor=(1.0 * u.MJy / u.sr)):
    """
    Create and populate MosaicModel of image and noises.

    Parameters
    ----------
    image : galsim.Image
        Image containing mosaic data.
    metadata : dict
        Metadata structure for Roman asdf file.
    efftimes : np.ndarray or float
        Effective exposure time of reach pixel in mosaic.
        If an array, shape must match shape parameter.
    rng : galsim.BaseDeviate
        Random number generator.
    seed : int
        Seed for random number generator.
    var_poisson : np.ndarray
        Poisson variance for each pixel
    var_flat : np.ndarray
        Flat variance for each pixel
    var_readnoise : np.ndarray
        Read Noise variance for each pixel
    context : np.ndarray
        File number(s) for each pixel
    unit_factor: float
        Factor to convert counts data to MJy / sr

    Returns
    -------
    objinfo : np.ndarray
        Information on position and flux of each rendered source.

    Returns
    -------
    image : rdm.MosaicModel
        Mosaic model object containing the data, metadata, variances, weight, and context.
    """

    # Create mosaic data object
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
    if var_poisson is None:
        mosaic_node.var_poisson = (unit_factor**2 / efftimes_arr**2).astype(np.float32)
    else:
        mosaic_node.var_poisson = u.Quantity(var_poisson.astype(np.float32), unit_factor.unit ** 2)

    # Read noise variance
    if var_readnoise is None:
        mosaic_node.var_rnoise = u.Quantity(np.zeros(mosaic.shape, dtype=np.float32), unit_factor.unit ** 2)
    else:
        mosaic_node.var_rnoise = u.Quantity((var_readnoise * np.ones(mosaic.shape)).astype(np.float32), unit_factor.unit ** 2)

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

    # Return mosaic
    return mosaic_node
