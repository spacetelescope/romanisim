"""Roman WFI simulator tool.

Based on galsim's implementation of Roman image simulation.  Uses galsim Roman modules
for most of the real work.
"""

import time
import os
import copy
import numpy as np
import astropy.time
from astropy import units as u, coordinates, table
import asdf
import galsim
from galsim import roman
from . import wcs, catalog, parameters, util, nonlinearity, ramp, log
import romanisim.l1
import romanisim.bandpass
import romanisim.psf
import romanisim.persistence

import roman_datamodels
import roman_datamodels.maker_utils as maker_utils


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

# should we let people specify reference files?
# what reference files do we use?
# distortion, read noise, dark, gain, flat
# these would each be optional arguments that would override


def make_l2(resultants, read_pattern, read_noise=None, gain=None, flat=None,
            linearity=None, darkrate=None, dq=None):
    """
    Simulate an image in a filter given resultants.

    This routine does idealized ramp fitting on a set of resultants.

    Parameters
    ----------
    resultants : np.ndarray[nresultants, ny, nx]
        resultants array
    read_pattern : list[list] (int)
        list of lists of indices of reads entering each resultant
    read_noise : np.ndarray[ny, nx] (float)
        read_noise image to use.  If None, use galsim.roman.read_noise.
    flat : np.ndarray[ny, nx] (float)
        flat field to use
    linearity : romanisim.nonlinearity.NL object or None
        non-linearity correction to use.
    darkrate : np.ndarray[ny, nx] (float)
        dark rate image to subtract from ramps (electron / s)
    dq : np.ndarray[nresultants, ny, nx] (int)
        DQ image corresponding to resultants

    Returns
    -------
    im : np.ndarray
        best fitting slopes
    var_rnoise : np.ndarray
        variance in slopes from read noise
    var_poisson : np.ndarray
        variance in slopes from source noise
    """

    if read_noise is None:
        read_noise = parameters.reference_data['readnoise']

    if gain is None:
        gain = parameters.reference_data['gain']
    try:
        gain = gain.astype('f4')
    except AttributeError:  # gain is not a Quantity
        gain = np.float32(gain)

    if linearity is not None:
        resultants = linearity.apply(resultants)

    log.info('Fitting ramps.')

    # commented out code below is inverse-covariance ramp fitting
    # which doesn't presently support DQ information
    # rampfitter = ramp.RampFitInterpolator(read_pattern)
    # ramppar, rampvar = rampfitter.fit_ramps(resultants * gain,
    #                                         read_noise * gain)

    if dq is None:
        dq = np.zeros(resultants.shape, dtype='i4')

    if linearity is not None:
        # Update data quality array for linearty coefficients
        dq |= linearity.dq

    ramppar, rampvar = ramp.fit_ramps_casertano(
        resultants * gain, dq & parameters.dq_do_not_use,
        read_noise * gain, read_pattern)
    # could iterate if we wanted to improve the flux estimates

    if darkrate is not None:
        ramppar[..., 1] -= darkrate

    if isinstance(gain, u.Quantity):
        gain = gain.value  # no values make sense except for electron / DN

    # The ramp fitter is not presently unit-aware; fix up the units by hand.
    # To do this right the ramp fitter should be made unit aware.
    # It takes a bit of work to get this right because we use the fact
    # that the variance of a Poisson distribution is equal to its mean,
    # which isn't true as soon as things start having units and requires
    # special handling.  And we use read_time without units a lot throughout
    # the code base.
    slopes = ramppar[..., 1] / gain * u.DN / u.s
    readvar = rampvar[..., 0, 1, 1] / gain**2 * (u.DN / u.s)**2
    poissonvar = rampvar[..., 1, 1, 1] / gain**2 * (u.DN / u.s)**2

    if flat is not None:
        flat = np.clip(flat, 1e-9, np.inf).astype('f4')
        slopes /= flat
        readvar /= flat**2
        poissonvar /= flat**2

    return slopes, readvar, poissonvar


def in_bounds(xx, yy, imbd, margin):
    """Filter sources to those landing near an image.

    Parameters
    ----------
    xx, yy: ndarray[nobj] (float)
        x & y positions of sources on image
    imbd : galsim.Image.Bounds
        bounds of image
    margin : int
        keep sources within this number of pixels of the image edge

    Returns
    -------
    keep : np.ndarray (bool)
        whether each source's center lands near the image (True) or not (False)
    """

    keep = ((xx > imbd.xmin - margin) & (xx < imbd.xmax + margin) & (
        yy > imbd.ymin - margin) & (yy < imbd.ymax + margin))
    return keep


def trim_objlist(objlist, image):
    """Trim objects down to those falling near an image.

    Objects must fall in a circle centered at the center of the image with
    radius 1.1 times the separation between the center and corner of the image.

    In contrast to in_bounds, this doesn't require the x and y coordinates of
    the individual sources, and just uses the source celestial coordinates
    directly.

    Parameters
    ----------
    objlist : astropy.table.Table including ra, dec columns
        Table of objects
    image : galsim.Image
        image near which objects should fall.

    Returns
    -------
    objlist : astropy.table.Table
        objlist trimmed to objects near image.
    """
    cc = coordinates.SkyCoord(
        objlist['ra'] * u.deg, objlist['dec'] * u.deg)
    center = image.wcs._radec(
        image.array.shape[0] // 2, image.array.shape[1] // 2)
    center = coordinates.SkyCoord(*np.array(center) * u.rad)
    corner = image.wcs._radec(0, 0)
    corner = coordinates.SkyCoord(*np.array(corner) * u.rad)
    sep = center.separation(cc)
    maxsep = 1.1 * corner.separation(center)  # 10% buffer
    keep = sep < maxsep
    objlist = objlist[keep]
    return objlist


def add_objects_to_image(image, objlist, xpos, ypos, psf,
                         flux_to_counts_factor, outputunit_to_electrons=None,
                         bandpass=None, filter_name=None, add_noise=False,
                         rng=None, seed=None):
    """Add sources to an image.

    Note: this includes Poisson noise when photon shooting is used
    (i.e., for chromatic source profiles), and otherwise is noise free, unless
    add_noise is set to True.

    Parameters
    ----------
    image : galsim.Image
        Image to which to add sources with associated WCS. Updated in place.
    objlist : list[CatalogObject]
        Objects to add to image.  These may be chromatic or achromatic.
    xpos, ypos : array_like
        x & y positions of sources (pixel) at which sources should be added
    psf : galsim.Profile
        PSF for image
    flux_to_counts_factor : float or list
        physical fluxes in objlist (whether in profile SEDs or flux arrays)
        should be multiplied by this factor to convert to total electrons in the
        image
    outputunit_to_electrons : array_like
        One output image unit corresponds to this many electrons.  If None,
        leave as electrons.
    bandpass : galsim.Bandpass
        bandpass in which image is being rendered.  This is used only in cases
        where chromatic profiles & PSFs are being used.
    filter_name : str
        filter to use to select appropriate flux from objlist.  This is only
        used when achromatic PSFs and sources are being rendered.
    add_noise : bool
        if True, add Poisson noise to noiseless FFT simulated images produced
        when achromatic profiles are used.
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    outinfo : np.ndarray
        Array structure containing rows for each source.  The columns give
        the total number of electrons from the source entering the image and
        the time taken to render the source.
    """
    if rng is None and seed is None:
        seed = 143
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    log.info(f'Adding {len(objlist)} sources to image...')
    nrender = 0

    chromatic = False
    if len(objlist) > 0 and objlist[0].profile.spectral:
        chromatic = True
    if len(objlist) > 0 and chromatic and bandpass is None:
        raise ValueError('bandpass must be set for chromatic PSF rendering.')
    if len(objlist) > 0 and not chromatic and filter_name is None:
        raise ValueError('must specify filter when using achromatic PSF '
                         'rendering.')

    outinfo = np.zeros(len(objlist), dtype=[('counts', 'f4'), ('time', 'f4')])
    for i, obj in enumerate(objlist):
        t0 = time.time()
        image_pos = galsim.PositionD(xpos[i], ypos[i])
        pwcs = image.wcs.local(image_pos)
        profile = obj.profile
        if not chromatic:
            if obj.flux is None:
                raise ValueError('Non-chromatic sources must have specified '
                                 'fluxes!')
            profile = profile.withFlux(obj.flux[filter_name])
        if hasattr(psf, 'at_position'):
            psf0 = psf.at_position(xpos[i], ypos[i])
        else:
            psf0 = psf
        factor = flux_to_counts_factor[i] if isinstance(flux_to_counts_factor, list) else flux_to_counts_factor
        final = galsim.Convolve(profile * factor, psf0)
        if chromatic:
            stamp = final.drawImage(
                bandpass, center=image_pos, wcs=pwcs,
                method='phot', rng=rng)
        else:
            try:
                stamp = final.drawImage(center=image_pos,
                                        wcs=pwcs)
                if add_noise:
                    stamp.addNoise(galsim.PoissonNoise(rng))
            except galsim.GalSimFFTSizeError:
                log.warning(f'Skipping source {i} due to too '
                            f'large FFT needed for desired accuracy.')
        if outputunit_to_electrons is not None:
            stamp /= outputunit_to_electrons[i]

        bounds = stamp.bounds & image.bounds
        if bounds.area() > 0:
            image[bounds] += stamp[bounds]
            counts = np.sum(stamp[bounds].array)
        else:
            counts = 0
        nrender += 1
        outinfo[i] = (counts, time.time() - t0)
    log.info('Rendered %d sources...' % nrender)
    return outinfo


def simulate_counts_generic(image, exptime, objlist=None, psf=None,
                            zpflux=None,
                            sky=None, dark=None,
                            flat=None, xpos=None, ypos=None,
                            ignore_distant_sources=10, bandpass=None,
                            filter_name=None, rng=None, seed=None,
                            **kwargs):
    """Add some simulated counts to an image.

    This routine intends to need to know nothing about Roman specifically.
    To do this, we need to have an image
    to start with with an attached WCS.  We also need an exposure time
    and potentially a zpflux so we know how to translate between the catalog
    fluxes and the electrons entering the image.  For chromatic rendering, this
    role is instead played by the bandpass, though the exposure time is still
    needed to handle that part of the conversion from flux to electrons.

    Then there are a few individual components that can be added on to
    an image:

    * objlist: a list of CatalogObjects to render, or a Table.  Can be chromatic
      or not.  This will have all your normal PSF and galaxy profiles.
    * sky: a sky background model.  This is different from a dark in that
      it is sensitive to the flat field.
    * dark: a dark model.
    * flat: a flat field for modulating the object and sky electrons

    Parameters
    ----------
    image : galsim.Image
        Image onto which other effects should be added, with associated WCS.
    exptime : float
        Exposure time
    objlist : list[CatalogObject], Table, or None
        Sources to render
    psf : galsim.Profile
        PSF to use when rendering sources
    zpflux : float
        For non-chromatic profiles, the factor converting flux to electrons / s.
    sky : float or array_like
        Image or constant with the electrons / pix / sec from sky.
    dark : float or array_like
        Image or constant with the electrons / pix / sec from dark current.
    flat : array_like
        Image giving the relative QE of different pixels.
    xpos, ypos : array_like (float)
        x, y positions of each source in objlist
    ignore_distant_sources : int
        Ignore sources more than this distance off image.
    bandpass : galsim.Bandpass
        bandpass to use for rendering chromatic objects
    filter_name : str
        name of filter (used to look up flux in achromatic case)
    rng : galsim.BaseDeviate
        random number generator
    seed : int
        seed for random number generator

    Returns
    -------
    objinfo : np.ndarray
        Information on position and flux of each rendered source.
    """
    if rng is None and seed is None:
        seed = 144
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)
    if (objlist is not None and len(objlist) > 0
            and image.wcs is None and (xpos is None or ypos is None)):
        raise ValueError('xpos and ypos must be set if rendering objects '
                         'without a WCS.')
    if objlist is None:
        objlist = []
    if len(objlist) > 0 and xpos is None:
        if isinstance(objlist, table.Table):
            objlist = trim_objlist(objlist, image)
            xpos, ypos = image.wcs._xy(
                np.radians(objlist['ra']), np.radians(objlist['dec']))
        else:
            coord = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                             for o in objlist])
            xpos, ypos = image.wcs._xy(coord[:, 0], coord[:, 1])
        # use private vectorized transformation
    if xpos is not None:
        xpos = np.array(xpos)
    if ypos is not None:
        ypos = np.array(ypos)
    if len(objlist) > 0:
        keep = in_bounds(xpos, ypos, image.bounds, ignore_distant_sources)
    else:
        keep = []
    if (len(objlist) > 0) and isinstance(objlist, table.Table):
        objlist = catalog.table_to_catalog(objlist[keep], [filter_name])
        xpos = xpos[keep]
        ypos = ypos[keep]
        keep = np.ones(len(objlist), dtype='bool')
    if len(objlist) > 0 and psf is None:
        raise ValueError('Must provide a PSF if you want to render objects.')

    if flat is None:
        flat = 1

    # for some reason, galsim doesn't like multiplying an SED by 1, but it's
    # okay with multiplying an SED by 1.0, so we cast to float.
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
    flux_to_counts_factor = exptime * maxflat
    if not chromatic:
        flux_to_counts_factor *= zpflux
    xposk = xpos[keep] if xpos is not None else None
    yposk = ypos[keep] if ypos is not None else None
    objinfokeep = add_objects_to_image(
        image, [o for (o, k) in zip(objlist, keep) if k],
        xposk, yposk, psf, flux_to_counts_factor,
        bandpass=bandpass, filter_name=filter_name, rng=rng)
    objinfo = np.zeros(
        len(objlist),
        dtype=[('x', 'f4'), ('y', 'f4'), ('counts', 'f4'), ('time', 'f4')])
    if len(objlist) > 0:
        objinfo['x'][keep] = xpos[keep]
        objinfo['y'][keep] = ypos[keep]
        objinfo['counts'][keep] = objinfokeep['counts']
        objinfo['time'][keep] = objinfokeep['time']

    # add Poisson noise if we made a noiseless, not-photon-shooting
    # image.
    poisson_noise = galsim.PoissonNoise(rng)
    if not chromatic:
        image.addNoise(poisson_noise)

    if sky is not None:
        workim = image * 0
        workim += sky * maxflat * exptime
        workim.addNoise(poisson_noise)
        image += workim

    if not np.all(flat == 1):
        image.quantize()
        rng_numpy_seed = rng.raw()
        rng_numpy = np.random.default_rng(rng_numpy_seed)
        image.array[:, :] = rng_numpy.binomial(
            image.array.astype('i4'), flat / maxflat)

    if dark is not None:
        workim = image * 0
        workim += dark * exptime
        workim.addNoise(poisson_noise)
        image += workim

    image.quantize()
    return objinfo


def simulate_counts(metadata, objlist,
                    rng=None, seed=None,
                    ignore_distant_sources=10, usecrds=True,
                    webbpsf=True,
                    darkrate=None, flat=None,
                    psf_keywords=dict()):
    """Simulate total electrons in a single SCA.

    This gives the total electrons recorded in an idealized instrument with no systematics;
    it includes only distortion & PSF convolution.  This total includes an appropriate amount
    of Poisson noise.

    Parameters
    ----------
    metadata : dict
        CRDS metadata dictionary
    objlist : list[CatalogObject] or Table
        Objects to simulate
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.
    ignore_distant_sources : float
        do not render sources more than this many pixels off edge of detector
    usecrds : bool
        use CRDS distortion map
    darkrate : float or np.ndarray[float]
        dark rate image to use (electrons / s)
    flat : float or np.ndarray[float]
        flat field to use
    psf_keywords : dict
        keywords passed to PSF generation routine

    Returns
    -------
    image : galsim.Image
        idealized image of scene as seen by Roman, giving total electron counts
        from rate sources (astronomical objects; backgrounds; dark current) in
        each pixel.
    simcatobj : np.ndarray
        catalog of simulated objects in image
    """

    read_pattern = metadata['exposure'].get(
        'read_pattern',
        parameters.read_pattern[metadata['exposure']['ma_table_number']])

    sca = int(metadata['instrument']['detector'][3:])
    exptime = parameters.read_time * read_pattern[-1][-1]
    if rng is None and seed is None:
        seed = 43
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    filter_name = metadata['instrument']['optical_element']

    date = metadata['exposure']['start_time']
    if not isinstance(date, astropy.time.Time):
        date = astropy.time.Time(date, format='isot')

    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter_name]
    bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]
    imwcs = wcs.get_wcs(metadata, usecrds=usecrds)
    chromatic = False
    if (len(objlist) > 0
            and not isinstance(objlist, table.Table)  # this case is always gray
            and objlist[0].profile.spectral):
        chromatic = True
    psf = romanisim.psf.make_psf(sca, filter_name, wcs=imwcs,
                                 chromatic=chromatic, webbpsf=webbpsf,
                                 variable=True, **psf_keywords)
    image = galsim.ImageF(roman.n_pix, roman.n_pix, wcs=imwcs, xmin=0, ymin=0)
    SCA_cent_pos = imwcs.toWorld(image.true_center)
    sky_level = roman.getSkyLevel(bandpass, world_pos=SCA_cent_pos,
                                  date=date.datetime, exptime=1)
    sky_level *= (1.0 + roman.stray_light_fraction)
    sky_image = image * 0
    imwcs.makeSkyImage(sky_image, sky_level)
    sky_image += roman.thermal_backgrounds[galsim_filter_name]
    abflux = romanisim.bandpass.get_abflux(filter_name)

    simcatobj = simulate_counts_generic(
        image, exptime, objlist=objlist, psf=psf, zpflux=abflux, sky=sky_image,
        dark=darkrate, flat=flat,
        ignore_distant_sources=ignore_distant_sources, bandpass=bandpass,
        filter_name=filter_name, rng=rng, seed=seed)

    return image, simcatobj


def gather_reference_data(image_mod, usecrds=False):
    """Gather reference data corresponding to metadata.

    This function pulls files from parameters.reference_data and/or
    CRDS to fill out the various reference files needed to perform
    the simulation.  If CRDS is set, values in parameters.reference_data
    are used instead of CRDS files when the reference_data are None.  If
    all CRDS files should be used, parameters.reference_data must contain
    only Nones.

    The input metadata is updated with CRDS software versions if CRDS
    is used.

    Returns
    -------
    dictionary containing the following keys:
        read_noise
        darkrate
        gain
        inv_linearity
        linearity
        saturation
        reffiles
    These have the reference images or constant values for the various
    reference parameters.
    """

    reffiles = {k: v for k, v in parameters.reference_data.items()}
    if 'photom' in reffiles:
        reffiles.pop('photom')

    out = dict(**reffiles)
    if usecrds:
        import crds
        refsneeded = [k for (k, v) in reffiles.items() if v is None]
        flatneeded = 'flat' in refsneeded
        if flatneeded:
            refsneeded.remove('flat')
        if len(refsneeded) > 0:
            reffiles.update(crds.getreferences(
                image_mod.get_crds_parameters(), reftypes=refsneeded,
                observatory='roman'))
            for reftype, reffn in reffiles.items():
                if reftype in ['inverselinearity', 'ipc', 'flat']:
                    continue
                if reftype not in refsneeded:
                    continue
                image_mod.meta.ref_file[reftype] = os.path.basename(reffn)
        if flatneeded:
            try:
                flatfile = crds.getreferences(
                    image_mod.get_crds_parameters(),
                    reftypes=['flat'], observatory='roman')['flat']

                flat_model = roman_datamodels.datamodels.FlatRefModel(flatfile)
                flat = flat_model.data[...].copy()
                image_mod.meta.ref_file['flat'] = os.path.basename(flatfile)
            except crds.core.exceptions.CrdsLookupError:
                log.warning('Could not find flat; using 1')
                flat = 1
            out['flat'] = flat
        image_mod.meta.ref_file.crds.version = crds.__version__
        image_mod.meta.ref_file.crds.context = crds.get_context_name(
            observatory=image_mod.crds_observatory)

    # reffiles has all of the reference files / values we know about

    nborder = parameters.nborder

    # we now need to extract the relevant fields
    if isinstance(reffiles['readnoise'], str):
        model = roman_datamodels.datamodels.ReadnoiseRefModel(
            reffiles['readnoise'])
        out['readnoise'] = model.data[nborder:-nborder, nborder:-nborder].copy()

    if isinstance(reffiles['gain'], str):
        model = roman_datamodels.datamodels.GainRefModel(reffiles['gain'])
        out['gain'] = model.data[nborder:-nborder, nborder:-nborder].copy()

    if isinstance(reffiles['dark'], str):
        model = roman_datamodels.datamodels.DarkRefModel(reffiles['dark'])
        out['dark'] = model.dark_slope[nborder:-nborder, nborder:-nborder].copy()
        out['dark'] *= out['gain']
    if isinstance(out['dark'], u.Quantity):
        out['dark'] = out['dark'].to(u.electron / u.s).value

    if isinstance(reffiles['inverselinearity'], str):
        ilin_model = roman_datamodels.datamodels.InverselinearityRefModel(
            reffiles['inverselinearity'])
        out['inverselinearity'] = nonlinearity.NL(
            ilin_model.coeffs[:, nborder:-nborder, nborder:-nborder].copy(),
            ilin_model.dq[nborder:-nborder, nborder:-nborder].copy(),
            gain=out['gain'])

    if isinstance(reffiles['linearity'], str):
        lin_model = roman_datamodels.datamodels.LinearityRefModel(
            reffiles['linearity'])
        out['linearity'] = nonlinearity.NL(
            lin_model.coeffs[:, nborder:-nborder, nborder:-nborder].copy(),
            lin_model.dq[nborder:-nborder, nborder:-nborder].copy(),
            gain=out['gain'])

    if isinstance(reffiles['saturation'], str):
        saturation = roman_datamodels.datamodels.SaturationRefModel(
            reffiles['saturation'])
        saturation = saturation.data[nborder:-nborder, nborder:-nborder].copy()
        out['saturation'] = saturation

    out['reffiles'] = reffiles
    return out


def simulate(metadata, objlist,
             usecrds=True, webbpsf=True, level=2, crparam=dict(),
             persistence=None, seed=None, rng=None,
             psf_keywords=dict(), **kwargs
             ):
    """Simulate a sequence of observations on a field in different bandpasses.

    Parameters
    ----------
    metadata : dict
        metadata structure for Roman asdf file, including information about

        * pointing: metadata['wcsinfo']['ra_ref'],
          metadata['wcsinfo']['dec_ref']
        * date: metadata['exposure']['start_time']
        * sca: metadata['instrument']['detector']
        * bandpass: metadata['instrument']['optical_detector']
        * ma_table_number: metadata['exposure']['ma_table_number']
        * read_pattern: metadata['exposure']['read_pattern']

    objlist : list[CatalogObject] or Table
        List of objects in the field to simulate
    usecrds : bool
        use CRDS to get reference files
    webbpsf : bool
        use webbpsf to generate PSF
    level : int
        0, 1 or 2, specifying level 1 or level 2 image
        0 makes a special idealized total electrons image; these are only
        intended for testing purposes and are not supported.
    persistence : romanisim.persistence.Persistence
        persistence object to use; None for no persistence
    crparam : dict
        Parameters for cosmic ray simulations.  None for no cosmic rays.
        Empty dictionary for default parameters.
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.
    psf_keywords : dict
        Keywords passed to the PSF generation routine

    Returns
    -------
    image : roman_datamodels model
        simulated image
    extras : dict
        Dictionary of additionally valuable quantities.  Includes at least
        simcatobj, the image positions and fluxes of simulated objects.  It may
        also include information on persistence and cosmic ray hits.
    """

    if not usecrds:
        log.warning('--usecrds is not set.  romanisim will not use reference '
                    'files from CRDS.  The WCS may be incorrect and up-to-date '
                    'calibration information will not be used.')

    meta = maker_utils.mk_common_meta()
    meta["photometry"] = maker_utils.mk_photometry()
    meta['wcs'] = None

    for key in parameters.default_parameters_dictionary.keys():
        meta[key].update(parameters.default_parameters_dictionary[key])

    for key in metadata.keys():
        meta[key].update(metadata[key])

    util.add_more_metadata(meta)

    # Create Image model to track validation
    image_node = maker_utils.mk_level2_image()
    image_node['meta'] = meta
    image_mod = roman_datamodels.datamodels.ImageModel(image_node)

    filter_name = image_mod.meta.instrument.optical_element

    read_pattern = metadata['exposure'].get(
        'read_pattern',
        parameters.read_pattern[metadata['exposure']['ma_table_number']])

    refdata = gather_reference_data(image_mod, usecrds=usecrds)
    read_noise = refdata['readnoise']
    darkrate = refdata['dark']
    gain = refdata['gain']
    inv_linearity = refdata['inverselinearity']
    linearity = refdata['linearity']
    saturation = refdata['saturation']
    reffiles = refdata['reffiles']
    flat = refdata['flat']
    pedestal_extra_noise = parameters.pedestal_extra_noise

    if rng is None and seed is None:
        seed = 43
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    if persistence is None:
        persistence = romanisim.persistence.Persistence()

    log.info('Simulating filter {0}...'.format(filter_name))
    counts, simcatobj = simulate_counts(
        image_mod.meta, objlist, rng=rng, usecrds=usecrds, darkrate=darkrate,
        webbpsf=webbpsf, flat=flat, psf_keywords=psf_keywords)
    util.update_pointing_and_wcsinfo_metadata(image_mod.meta, counts.wcs)
    if level == 0:
        im = dict(data=counts.array, meta=dict(image_mod.meta.items()))
    else:
        l1, l1dq = romanisim.l1.make_l1(
            counts, read_pattern, read_noise=read_noise,
            pedestal_extra_noise=pedestal_extra_noise,
            rng=rng, gain=gain,
            crparam=crparam,
            inv_linearity=inv_linearity,
            tstart=image_mod.meta.exposure.start_time,
            persistence=persistence,
            saturation=saturation,
            **kwargs)
    if level == 1:
        im, extras = romanisim.l1.make_asdf(
            l1, dq=l1dq, metadata=image_mod.meta, persistence=persistence)
    elif level == 2:
        slopeinfo = make_l2(l1, read_pattern, read_noise=read_noise,
                            gain=gain, flat=flat, linearity=linearity,
                            darkrate=darkrate, dq=l1dq)
        l2dq = np.bitwise_or.reduce(l1dq, axis=0)
        im, extras = make_asdf(
            *slopeinfo, metadata=image_mod.meta, persistence=persistence,
            dq=l2dq, imwcs=counts.wcs, gain=gain)
    else:
        extras = dict()

    if reffiles:
        extras["simulate_reffiles"] = {}
        for key, value in reffiles.items():
            extras["simulate_reffiles"][key] = value

    extras['simcatobj'] = simcatobj
    extras['wcs'] = wcs.convert_wcs_to_gwcs(counts.wcs)
    log.info('Simulation complete.')
    return im, extras


def make_test_catalog_and_images(
        seed=12345, sca=7, filters=None, nobj=1000,
        usecrds=True, webbpsf=True, galaxy_sample_file_name=None, **kwargs):
    """This is a test routine that exercises many options but is not intended for
    general use."""
    log.info('Making catalog...')
    if filters is None:
        filters = ['Y106', 'J129', 'H158']
    metadata = copy.deepcopy(parameters.default_parameters_dictionary)
    metadata['instrument']['detector'] = 'WFI%02d' % sca
    imwcs = wcs.get_wcs(metadata, usecrds=usecrds)
    rd_sca = imwcs.toWorld(galsim.PositionD(
        roman.n_pix / 2 + 0.5, roman.n_pix / 2 + 0.5))
    cat = catalog.make_dummy_catalog(
        rd_sca, seed=seed, nobj=nobj,
        galaxy_sample_file_name=galaxy_sample_file_name)
    rng = galsim.UniformDeviate(0)
    out = dict()
    for filter_name in filters:
        metadata['instrument']['optical_element'] = 'F' + filter_name[1:]
        im = simulate(metadata, objlist=cat, rng=rng, usecrds=usecrds,
                      webbpsf=webbpsf, **kwargs)
        out[filter_name] = im
    return out


def make_asdf(slope, slopevar_rn, slopevar_poisson, metadata=None,
              filepath=None, persistence=None, dq=None, imwcs=None,
              gain=None):
    """Wrap a galsim simulated image with ASDF/roman_datamodel metadata.
    """

    out = maker_utils.mk_level2_image(
        n_groups=len(metadata['exposure']['read_pattern']))
    # ephemeris contains a lot of angles that could be computed.
    # exposure contains
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
    # visit: start_time, end_time, total_exposures, ...?
    # wcsinfo: v2_ref, v3_ref, vparity, v3yangle, ra_ref, dec_ref
    #     roll_ref, s_region
    if metadata is not None:
        out['meta'].update(metadata)

    if imwcs is not None:  # add a WCS
        gwcs = wcs.convert_wcs_to_gwcs(imwcs)
        out['meta'].update(wcs=gwcs)
        out['meta']['wcsinfo']['s_region'] = wcs.create_s_region(gwcs)

    util.update_photom_keywords(out, gain=gain)

    out['data'] = slope.value
    out['dq'] = np.zeros(slope.shape, dtype='u4')
    if dq is not None:
        out['dq'][:, :] = dq
    out['var_poisson'] = slopevar_poisson.value
    out['var_rnoise'] = slopevar_rn.value
    out['var_flat'] = slopevar_rn.value * 0
    out['err'] = np.sqrt(out['var_poisson'] + out['var_rnoise'] + out['var_flat'])
    extras = dict()
    if persistence is not None:
        extras['persistence'] = persistence.to_dict()
    if filepath:
        af = asdf.AsdfFile()
        af.tree = {'roman': out, 'romanisim': extras}
        af.write_to(filepath)
    return out, extras


def inject_sources_into_l2(model, cat, x=None, y=None, psf=None, rng=None,
                           gain=None, webbpsf=True):
    """Inject sources into an L2 image.

    This routine allows sources to be injected into an existing L2 image.
    Source injection into an L2 image relies on knowing the objects'
    x and y locations, the PSF, and the image gain; if these are not provided,
    reasonable defaults are generated from the input model.

    The simulation proceeds by (optionally) using the model WCS to generate the
    x & y locations, grabbing the gain from
    romanisim.parameters.reference_data, and grabbing the read_pattern from the
    model_metadata.  The number of additional counts in each pixel are
    simulated.  We create a "virtual" ramp that uses the input L2 image and
    evenly apportions the measured DN/s along the ramp using the MA table.  We
    apportion the new counts to a new ramp, and add the new ramp to the virtual
    ramp.  We then refit the new ramp, and replace the old fit with the new
    fit.

    This simulation is not as complete as the full L2 simulation.  We do not
    include non-linearity or saturation, for example.  Identified CR hits
    are lost.  But it should do a decent job at providing realistic
    uncertainties otherwise, including how read & Poisson noise influence
    the choice of weights used in ramp fitting and how those influence the
    final uncertainties.

    Parameters
    ----------
    model: roman_datamodels.datamodels.ImageModel
        model into which to inject sources
    cat: astropy.table.Table
        catalog of sources to inject into image
    x: list[float] or None
        x coordinates of catalog locations in image
    y: list[float] or None
        y coordinates of catalog locations in image
    psf: galsim.gsobject.GSObject
        PSF to use
    rng: galsim.BaseDeviate
        galsim random number generator to use
    gain: float [electron / DN]
        gain to use when converting simulated electrons to DN
    webbpsf: bool
        if True, use WebbPSF to model the PSF

    Returns
    -------
    model_out: roman_datamodels.datamodel.ImageModel
        model with additional sources
    """
    if rng is None:
        rng = galsim.UniformDeviate(123)

    if x is None or y is None:
        x, y = model.meta.wcs.numerical_inverse(cat['ra'], cat['dec'],
                                                with_bounding_box=False)

    filter_name = model.meta.instrument.optical_element
    cat = catalog.table_to_catalog(cat, [filter_name])

    # are we doing photon shooting?
    chromatic = False
    if len(cat) > 0 and cat[0].profile.spectral:
        chromatic = True

    wcs = romanisim.wcs.GWCS(model.meta.wcs)
    if psf is None:
        psf = romanisim.psf.make_psf(
            int(model.meta.instrument.detector[-2:]), filter_name, wcs=wcs,
            chromatic=False, webbpsf=webbpsf)

    if gain is None:
        gain = parameters.reference_data['gain']

    # assemble bits we need in order to add a source to an image
    sourcecounts = galsim.ImageF(model.data.shape[0], model.data.shape[1],
                                 wcs=wcs, xmin=0, ymin=0)
    galsim_filter_name = romanisim.bandpass.roman2galsim_bandpass[filter_name]
    bandpass = roman.getBandpasses(AB_zeropoint=True)[galsim_filter_name]
    abflux = romanisim.bandpass.get_abflux(filter_name)
    read_pattern = model.meta.exposure.read_pattern
    exptime = parameters.read_time * read_pattern[-1][-1]
    tij = romanisim.l1.read_pattern_to_tij(read_pattern)
    tbar = ramp.read_pattern_to_tbar(read_pattern)
    flux_to_counts_factor = exptime
    if not chromatic:
        flux_to_counts_factor *= abflux

    # compute the total number of counts we got from the source
    add_objects_to_image(
        sourcecounts, cat,
        xpos=x, ypos=y, psf=psf,
        flux_to_counts_factor=flux_to_counts_factor,
        bandpass=bandpass, filter_name=filter_name, rng=rng)

    m = sourcecounts.array != 0

    # add Poisson noise if we made a noiseless, not-photon-shooting
    # image.
    if not chromatic:
        pd = galsim.PoissonDeviate(rng)
        noise = np.clip(sourcecounts.array[m].copy(), 0, np.inf)
        pd.generate_from_expectation(noise)
        sourcecounts.array[m] = noise

    sourcecounts.quantize()

    m = sourcecounts.array != 0
    # many pixels which may have received a small fraction of a count
    # receive 0 after the Poisson sampling.

    # create injected source ramp resultants
    resultants, dq = romanisim.l1.apportion_counts_to_resultants(
        sourcecounts.array[m], tij, rng=rng)
    resultants = resultants * u.electron

    # Inject source to original image
    newramp = model.data[None, :] * tbar[:, None, None] * u.DN
    newramp[:, m] += resultants / gain
    # newramp has units of DN

    # Make new image of the combination
    newimage, readvar, poissonvar = make_l2(
        newramp[:, m], read_pattern,
        gain=gain, flat=1, darkrate=0)

    res = copy.deepcopy(model)
    res.data[m] = newimage
    res.var_rnoise[m] = readvar
    res.var_poisson[m] = poissonvar
    res.err[m] = np.sqrt(readvar + poissonvar)
    return res
