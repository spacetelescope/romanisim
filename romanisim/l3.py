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


def add_objects_to_l3(l3_mos, source_cat, exptimes, xpos, ypos, psf,
                      etomjysr=1.0, maggytoes=1.0,
                      filter_name=None, bandpass=None,
                      rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_mos : MosaicModel or galsim.Image
        Mosaic of images
    source_cat : list[CatalogObject]
        List of catalog objects to add to l3_mos
    exptimes : list
        Exposure times to scale back to rate units
    xpos, ypos : array_like
        x & y positions of sources (pixel) at which sources should be added
    coords : array_like
        ra & dec positions of sources (deg) at which sources should be added
    maggytoes : float
        Factor to convert maggies to e/s
    etomjysr: float
        Factor to convert electrons to MJy / sr
    filter_name : str
        Filter to use to select appropriate flux from objlist. This is only
        used when achromatic PSFs and sources are being rendered.
    bandpass : galsim.Bandpass
        Bandpass in which mosaic is being rendered. This is used only in cases
        where chromatic profiles & PSFs are being used.
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    Information from romanisim.image.add_objects_to_image.  Note
    that l3_mos is updated in place.
    """
    # Obtain optical element
    if filter_name is None:
        filter_name = l3_mos.meta.basic.optical_element

    # Create Image canvas to add objects to
    if isinstance(l3_mos, (rdm.MosaicModel, WfiMosaic)):
        sourcecountsall = galsim.ImageF(
            l3_mos.data.value, wcs=romanisim.wcs.GWCS(l3_mos.meta.wcs),
            xmin=0, ymin=0)
    else:
        sourcecountsall = l3_mos

    # Add sources to the original mosaic data array
    outinfo = romanisim.image.add_objects_to_image(
        sourcecountsall, source_cat, xpos, ypos,
        psf, flux_to_counts_factor=[xpt * maggytoes for xpt in exptimes],
        outputunit_to_electrons=[xpt / etomjysr for xpt in exptimes],
        bandpass=bandpass, filter_name=filter_name,
        rng=rng, seed=seed, add_noise=True)

    # Save array with added sources
    if isinstance(l3_mos, (rdm.MosaicModel, WfiMosaic)):
        l3_mos.data = sourcecountsall.array * l3_mos.data.unit

    return outinfo


def inject_sources_into_l3(model, cat, x=None, y=None, psf=None, rng=None,
                           gain=None, webbpsf=True, exptimes=None, seed=None):
    """Inject sources into an L3 image.

    This routine allows sources to be injected onto an existing L3 image.
    Source injection into an L3 image relies on knowing the objects'
    x and y locations, the PSF, and the exposure time; if these are not
    provided, reasonable defaults are generated.

    The simulation proceeds by (optionally) using the model WCS to generate the
    x & y locations.  It also optionally computes an appropriate pixel-convolved
    PSF for the image.  It optionally uses the model Poisson variances and
    model fluxes to estimate the per-source exposure times appropriate
    for injecting the sources in the catalog onto the image.  Finally,
    it updates the var_poisson of the input image to account for the additional
    variance from the added sources.

    Parameters
    ----------
    model: roman_datamodels.datamodels.WfiMosaic
        model into which to inject sources
    cat: astropy.table.Table
        catalog of sources to inject into image
    x: list[float] or None
        x coordinates of catalog locations in image
    y: list[float] or None
        y coordinates of catalog locations in image
    exptimes : list[float] or None
        exposure times of each source.  Computed from model.var_poisson
        and model.flux at each source location if not specified.
    psf: galsim.gsobject.GSObject
        PSF to use
    rng: galsim.BaseDeviate
        galsim random number generator to use
    seed : int
        Seed to use for rng
    webbpsf: bool
        if True, use WebbPSF to model the PSF

    Returns
    -------
    outinfo : np.ndarray with information about added sources
    """
    if seed is None:
        seed = 125
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    if x is None or y is None:
        x, y = model.meta.wcs.numerical_inverse(cat['ra'], cat['dec'],
                                                with_bounding_box=False)

    filter_name = model.meta.basic.optical_element
    cat = romanisim.catalog.table_to_catalog(cat, [filter_name])

    wcs = romanisim.wcs.GWCS(model.meta.wcs)
    if psf is None:
        pixscalefrac = get_pixscalefrac(wcs, model.data.shape)
        if (pixscalefrac > 1) or (pixscalefrac < 0):
            raise ValueError('weird pixscale!')
        psf = l3_psf(filter_name, pixscalefrac, webbpsf=True, chromatic=False)

    maggytoes = romanisim.bandpass.get_abflux(filter_name)
    etomjysr = romanisim.bandpass.etomjysr(filter_name)

    Ct = []
    for idx, (x0, y0) in enumerate(zip(x, y)):
        # Set scaling factor for injected sources
        # Flux / sigma_p^2
        xidx, yidx = int(np.round(x0)), int(np.round(y0))
        if model.var_poisson[yidx, xidx].value != 0:
            Ct.append(math.fabs(
                model.data[yidx, xidx].value /
                model.var_poisson[yidx, xidx].value))
        else:
            Ct.append(1.0)
    Ct = np.array(Ct)
    # etomjysr = 1/C; C converts fluxes to electrons
    exptimes = Ct * etomjysr

    Ct_all = (model.data.value /
              (model.var_poisson.value + (model.var_poisson.value == 0)))

    # compute the total number of counts we got from the source
    res = add_objects_to_l3(
        model, cat, exptimes, x, y, psf, etomjysr=etomjysr,
        maggytoes=maggytoes, filter_name=filter_name, bandpass=None,
        rng=rng)

    model.var_poisson = (model.data.value / Ct_all) * model.var_poisson.unit

    return res



def generate_exptime_array(cat, meta):
    """ To be updated.
    Function to ascertain / set exposure time in each pixel
    Present code is placeholder to be built upon
    """

    # Get wcs for this metadata
    twcs = romanisim.wcs.GWCS(romanisim.wcs.get_mosaic_wcs(meta))

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


def l3_psf(bandpass, scale=0, chromatic=False, **kw):
    """Construct a PSF for an L3 image.

    The challenge here is that the L3 PSF has gone through drizzling which
    broadens the PSF relative to the native PSF.  We treat this here by doing
    the following:

    * we start with the native PSF for the appropriate bandpass for SCA 2
      (no pixel convolution)
    * we convolve that with a box of size sqrt(1-scale**2) pixel
    * we return the result

    So for a very small scale you still get a convolution by the full
    native pixel scaling (since the image was ultimately observed with the
    Roman detectors and needs to be seen through that grid), but you get
    oversampled output.  There will
    always be an additional convolution with the output pixel scale.  In
    the limit that the output pixel scale is 1, this does nothing, but
    provides undersampled output.

    Extra arguments are passed to romanisim.psf.make_psf .

    Parameters
    ----------
    pixscale : float
        The output mosaic pixel scale.  Must be between 0 and 1.
    bandpass : str
        The filter to use

    Returns
    -------
    galsim.Profile object for use as the PSF
    """

    if scale < 0 or scale > 1:
        raise ValueError('scale must be between 0 and 1')
    convscale = romanisim.parameters.pixel_scale * np.sqrt(
        1 - scale**2)
    if scale == 1:
        extra_convolution = None
    else:
        extra_convolution = galsim.Pixel(
            convscale * romanisim.parameters.pixel_scale)
    psf = romanisim.psf.make_psf(filter_name=bandpass,
                                 sca=romanisim.parameters.default_sca,
                                 extra_convolution=extra_convolution,
                                 chromatic=chromatic, **kw)
    return psf


def simulate(shape, wcs, efftimes, filter_name, catalog, nexposures=1,
             metadata={}, 
             effreadnoise=None, sky=None, psf=None,
             bandpass=None, seed=None, rng=None, webbpsf=True,
             **kwargs):
    """Simulate a sequence of observations on a field in different bandpasses.

    Parameters
    ----------
    shape : tuple
        Array shape of mosaic to simulate.
    wcs : gwcs.wcs.WCS
        WCS corresponding to image.  Will only work well with square pixels.
    efftimes : np.ndarray or float
        Effective exposure time of reach pixel in mosaic.
        If an array, shape must match shape parameter.
    filter_name : str
        Filter to use to select appropriate flux from objlist. This is only
        used when achromatic PSFs and sources are being rendered.
    catalog : list[CatalogObject] or Table
        List of catalog objects to add to l3_mos
    nexposures : int
        Number of exposures on the field.  Used to compute effreadnoise
        estimate if not explicitly set, otherwise not used.
    metadata : dict
        Metadata structure for Roman asdf file.
    effreadnoise : float
        Effective read noise for mosaic (MJy / sr)
    sky : float or array_like
        Image or constant with sky and other backgrounds (MJy / sr).  If None, then
        sky will be generated from galsim's getSkyLevel for Roman for the
        date provided in metadata[basic][time_mean_mjd].
    psf : galsim.Profile or None
        PSF for image
    bandpass : galsim.Bandpass
        Bandpass in which mosaic is being rendered. This is used only in cases
        where chromatic profiles & PSFs are being used.
    webbpsf : bool
        Use webbpsf to compute PSF
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
    meta['basic']['optical_element'] = filter_name
    for key in metadata.keys():
        if key not in meta:
            meta[key] = metadata[key]
        else:
            meta[key].update(metadata[key])
    # May need an equivalent of this this for L3?
    # util.add_more_metadata(meta)

    log.info('Simulating filter {0}...'.format(filter_name))

    # Get filter and bandpass
    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter_name]
    if bandpass is None:
        bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]

    # Create initial galsim image
    image = galsim.ImageF(shape[0], shape[1], wcs=romanisim.wcs.GWCS(wcs),
                          xmin=0, ymin=0)

    etomjysr = romanisim.bandpass.etomjysr(filter_name)

    # Create sky for this mosaic, if not provided (in cps)
    if sky is None:
        date = meta['basic']['time_mean_mjd']
        if not isinstance(date, astropy.time.Time):
            date = astropy.time.Time(date, format='mjd')

        # Examine this in more detail to ensure it is correct
        # Create base sky level
        mos_cent_pos = image.wcs.toWorld(image.true_center)
        sky_level = roman.getSkyLevel(bandpass, world_pos=mos_cent_pos, exptime=1)
        sky_level *= (1.0 + roman.stray_light_fraction)
        sky_mosaic = image * 0
        image.wcs.makeSkyImage(sky_mosaic, sky_level)
        sky_mosaic += roman.thermal_backgrounds[galsim_filter_name]
        sky = sky_mosaic
    else:
        sky = sky / etomjysr  # convert to electrons / s / nominal pix
        # note that we might be making different pixel scales in the
        # output, with correspondingly less sky.  We choose to see this
        # as those pixels having lower exposure time, and keep this in
        # units of electron / s / nominal pixel.

    # Flux in AB mags to electrons
    maggytoes = romanisim.bandpass.get_abflux(filter_name)

    # Set effective read noise
    if effreadnoise is None:
        readnoise = np.median(romanisim.parameters.reference_data['readnoise'])
        gain = np.median(romanisim.parameters.reference_data['gain'])
        effreadnoise = (
            np.sqrt(2) * readnoise * gain)
        # sqrt(2) from subtracting one read from another
        effreadnoise /= (np.median(efftimes) / nexposures)
        # divided by the typical exposure length
        effreadnoise /= np.sqrt(nexposures)
        # averaging down like the sqrt of the number of exposures
        # note that we are ignoring all of the individual reads, which also
        # contribute to reducing the effective read noise.  Pass --effreadnoise
        # if you want to do better than this!
        effreadnoise = effreadnoise.to(u.electron).value * etomjysr
        # converting to MJy/sr units
    else:
        effreadnoise = 0

    chromatic = False
    if (len(catalog) > 0 and not isinstance(catalog, astropy.table.Table)
            and catalog[0].profile.spectral):
        chromatic = True

    if psf is None:
        pixscalefrac = get_pixscalefrac(image.wcs, shape)
        if (pixscalefrac > 1) or (pixscalefrac < 0):
            raise ValueError('weird pixscale!')
        psf = l3_psf(filter_name, pixscalefrac, webbpsf=webbpsf,
                     chromatic=chromatic)

    # Simulate mosaic cps
    mosaic, extras = simulate_cps(
        image, filter_name, efftimes, objlist=catalog, psf=psf, 
        sky=sky,
        effreadnoise=effreadnoise, bandpass=bandpass,
        rng=rng, seed=seed,
        maggytoes=maggytoes, etomjysr=etomjysr,
        **kwargs)

    # Create Mosaic Model
    var_poisson = extras.pop('var_poisson')
    var_rnoise = extras.pop('var_rnoise')
    context = np.ones((1,) + mosaic.array.shape, dtype=np.uint32)
    mosaic_mdl = make_l3(mosaic, metadata, efftimes, var_poisson=var_poisson,
                         var_rnoise=var_rnoise, context=context)

    log.info('Simulation complete.')
    # return mosaic, extras
    return mosaic_mdl, extras


def get_pixscalefrac(wcs, shape):
    """Compute pixel scale from WCS, scaled to nominal units of 0.11".

    Computes the difference in arcseconds between the central pixels.
    Assumes square pixels.  Scales this so that a nominal WCS with 0.11"
    pixels gives a value of 1.

    Parameters
    ----------
    wcs : galsim.WCS
        WCS to get pixel scale for
    shape : tuple
        image shape
    """
    cenpix = shape[0] // 2, shape[1] // 2
    pos1 = wcs.toWorld(galsim.PositionD(cenpix[0], cenpix[1]))
    pos2 = wcs.toWorld(galsim.PositionD(cenpix[0], (cenpix[1] + 1)))
    pixscale = pos1.distanceTo(pos2).deg * 60 * 60  # arcsec
    pixscale /= romanisim.parameters.pixel_scale
    if (pixscale > 1 and pixscale < 1.05):
        pixscale = 1
        log.info('Setting pixscale to match default.')
    return pixscale


def simulate_cps(image, filter_name, efftimes, objlist=None, psf=None,
                 xpos=None, ypos=None, coord=None, sky=0, bandpass=None,
                 effreadnoise=None, maggytoes=None, etomjysr=None,
                 rng=None, seed=None, ignore_distant_sources=10,):
    """Simulate average MegaJankies per steradian in a single SCA.

    Parameters
    ----------
    image : galsim.Image
        Image onto which other effects should be added, with associated WCS.
    filter_name : str
        filter to simulate
    efftimes : np.ndarray or float
        Effective exposure time of reach pixel in mosaic.
        If an array, shape must match shape parameter.
    objlist : list[CatalogObject], Table, or None
        Sources to render
    psf : galsim.Profile
        PSF to use when rendering sources
    xpos, ypos : array_like (float)
        x, y positions of each source in objlist
    coord : array_like (float)
        ra, dec positions of each source in objlist (deg)
    sky : float or array_like
        Image or constant with the electron / pix / sec from sky.
    bandpass : galsim.Bandpass
        bandpass being used.  Only used for chromatic objects
    effreadnoise : float
        Effective read noise for mosaic (MJy / sr)
    maggytoes: float
        Factor to convert electrons to MJy / sr; one maggy makes
        this many e/s.
    etomjysr : float
        Factor to convert electron to MJy/sr;  one e/s/pix corresponds
        to this MJy/sr.
    rng : galsim.BaseDeviate
        random number generator
    seed : int
        seed for random number generator
    ignore_distant_sources : int
        Ignore sources more than this distance off image.

    Returns
    -------
    image : galsim.Image
        Idealized image of scene as seen by Roman (MJy / sr)
    extras : dict
        catalog of simulated objects in image, noise, and misc. debug
    """

    if rng is None and seed is None:
        seed = 144
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    if etomjysr is None:
        etomjysr = romanisim.bandpass.etomjysr(filter_name)

    if maggytoes is None:
        maggytoes = romanisim.bandpass.get_abflux(filter_name)

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
            coord = np.array([[o['ra'], o['dec']] for o in objlist])
        else:
            coord = np.array([[o.sky_pos.ra.deg, o.sky_pos.dec.deg]
                             for o in objlist])
        xpos, ypos = image.wcs.radecToxy(coord[:, 0], coord[:, 1], 'deg')

    if xpos is not None:
        xpos = np.array(xpos)
    if ypos is not None:
        ypos = np.array(ypos)

    # Check for objects outside the image boundary (+ consideration)
    if len(objlist) > 0:
        keep = romanisim.image.in_bounds(xpos, ypos, image.bounds,
                                         ignore_distant_sources)

        if isinstance(objlist, astropy.table.Table):
            objlist = objlist[keep]
        else:
            objlist = [o for (o, k) in zip(objlist, keep) if k]
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

    if isinstance(objlist, astropy.table.Table):
        objlist = romanisim.catalog.table_to_catalog(objlist, [filter_name])

    # Add objects to mosaic
    if len(src_exptimes) > 0:
        maggytoes0 = maggytoes if not objlist[0].profile.spectral else 1
        objinfokeep = add_objects_to_l3(
            image, objlist, src_exptimes, xpos=xpos, ypos=ypos,
            filter_name=filter_name, psf=psf, bandpass=bandpass, rng=rng,
            maggytoes=maggytoes0, etomjysr=etomjysr)

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

    if sky is not None:
        # in e / s
        poisson_noise = galsim.PoissonNoise(rng)
        workim = image * 0
        workim += sky * efftimes
        workim.addNoise(poisson_noise)
        image += (workim * etomjysr / efftimes)

    # Add readnoise
    if effreadnoise is not None:
        # in MJy / sr
        readnoise = np.zeros(image.array.shape, dtype='f4')
        rn_rng = galsim.GaussianDeviate(seed)
        rn_rng.generate(readnoise)
        readnoise = readnoise * effreadnoise
        image += readnoise
    else:
        effreadnoise = 0
    extras['var_rnoise'] = effreadnoise

    var_poisson_factor = (efftimes / etomjysr) * etomjysr ** 2 / efftimes ** 2
    # goofy game with etomjysr: image * (efftimes / etomjysr * efftimes)
    # -> number of photons entering each pixel
    # then we interpret this as a variance (since mean = variance for a
    # Poisson distribution), and convert the variance image
    # to the final units with two factors of etomjysr and the effective time

    extras['var_poisson'] = (
        np.clip(image.array, 0, np.inf) /
        etomjysr * efftimes * etomjysr ** 2 / efftimes ** 2)

    # Return image and artifacts
    return image, extras


def make_l3(image, metadata, efftimes, var_poisson=None,
            var_flat=None, var_rnoise=None, context=None):
    """
    Create and populate MosaicModel of image and noises.

    Parameters
    ----------
    image : galsim.Image
        Image containing mosaic data (MJy / sr)
    metadata : dict
        Metadata structure for Roman asdf file.
    efftimes : np.ndarray or float
        Effective exposure time of reach pixel in mosaic.
        If an array, shape must match shape parameter.
    var_poisson : np.ndarray
        Poisson variance for each pixel
    var_flat : np.ndarray
        Flat variance for each pixel
    var_rnoise : np.ndarray
        Read Noise variance for each pixel
    context : np.ndarray
        File number(s) for each pixel

    Returns
    -------
    image : roman_datamodels.datamodels.MosaicModel
        Mosaic datamodel
    """

    # Create mosaic data object
    mosaic = image.array.copy() * u.MJy / u.sr

    # Ensure that effective times are an array
    if isinstance(efftimes, np.ndarray):
        efftimes_arr = efftimes
    else:
        efftimes_arr = efftimes * np.ones(mosaic.shape, dtype=np.float32)

    # Set mosaic to be a mosaic node
    mosaic_node = maker_utils.mk_level3_mosaic(
        shape=mosaic.shape, metadata=metadata)

    # Set data
    mosaic_node.data = mosaic

    var_poisson = 0 if var_poisson is None else var_poisson
    var_rnoise = 0 if var_rnoise is None else var_rnoise
    var_flat = 0 if var_flat is None else var_flat
    context = (np.ones((1,) + mosaic.shape, dtype=np.uint32)
               if context is None else context)
    
    mosaic_node.var_poisson[...] = var_poisson * mosaic.unit ** 2
    mosaic_node.var_rnoise[...] = var_rnoise * mosaic.unit ** 2
    mosaic_node.var_flat[...] = var_flat * mosaic.unit ** 2
    mosaic_node.err[...] = np.sqrt(
        var_poisson + var_rnoise + var_flat) * mosaic.unit
    mosaic_node.context = context
    

    # Weight
    # Use exptime weight
    mosaic_node.weight = efftimes_arr.astype(np.float32)

    return mosaic_node
