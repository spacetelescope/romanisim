"""Roman WFI simulator tool.

Based on galsim's implementation of Roman image simulation.  Uses galsim Roman modules
for most of the real work.
"""

import time
import copy
import numpy as np
import astropy.time
from astropy import units as u
from astropy import coordinates
from astropy import table
import asdf
import galsim
from galsim import roman


from . import wcs
from . import catalog
from . import parameters
from . import util
from . import nonlinearity
import romanisim.l1
import romanisim.bandpass
import romanisim.psf
import romanisim.persistence
from romanisim import log

import roman_datamodels
try:
    import roman_datamodels.maker_utils as maker_utils
except ImportError:
    import roman_datamodels.testing.utils as maker_utils


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


def make_l2(resultants, ma_table, read_noise=None, gain=None, flat=None,
            linearity=None, dark=None, dq=None):
    """
    Simulate an image in a filter given resultants.

    This routine does idealized ramp fitting on a set of resultants.

    Parameters
    ----------
    resultants : np.ndarray[nresultants, nx, ny]
        resultants array
    ma_table : list[list] (int)
        list of list of first read numbers and number of reads in each resultant
    read_noise : np.ndarray[nx, ny] (float)
        read_noise image to use.  If None, use galsim.roman.read_noise.
    flat : np.ndarray[nx, ny] (float)
        flat field to use
    linearity : romanisim.nonlinearity.NL object or None
        non-linearity correction to use.
    dark : np.ndarray[nresultants, nx, ny] (float)
        dark image to subtract from ramps (DN)
    dq : np.ndarray[nresultants, nx, ny] (int)
        DQ image corresponding to resultants

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
        read_noise = parameters.read_noise

    if gain is None:
        gain = parameters.gain

    if linearity is not None:
        resultants = linearity.apply(resultants)

    # no error propagation

    if dark is not None:
        resultants = resultants - dark

    from . import ramp
    log.info('Fitting ramps.')

    # commented out code below is inverse-covariance ramp fitting
    # which doesn't presently support DQ information
    # rampfitter = ramp.RampFitInterpolator(ma_table)
    # ramppar, rampvar = rampfitter.fit_ramps(resultants * gain,
    #                                         read_noise * gain)

    if dq is None:
        dq = np.zeros(resultants.shape, dtype='i4')

    if linearity is not None:
        # Update data quality array for linearty coefficients
        dq = np.bitwise_or(dq, linearity.dq)

    ramppar, rampvar = ramp.fit_ramps_casertano(resultants * gain, dq,
                                                read_noise * gain, ma_table)

    log.warning('The ramp fitter is unaware of noise from dark current because '
                'it runs on dark-subtracted images.  We could consider adding '
                'a separate dark rate term to fit ramps.')
    # could iterate if we wanted to improve the flux estimates

    # The ramp fitter is not presently unit-aware; fix up the units by hand.
    # To do this right the ramp fitter should be made unit aware.
    # It takes a bit of work to get this right because we use the fact
    # that the variance of a Poisson distribution is equal to its mean,
    # which isn't true as soon as things start having units and requires
    # special handling.  And we use read_time without units a lot throughout
    # the code base.
    slopes = ramppar[..., 1] / u.s
    readvar = rampvar[..., 0, 1, 1] * (u.electron / u.s)**2
    poissonvar = rampvar[..., 1, 1, 1] * (u.electron / u.s)**2

    if flat is not None:
        flat = np.clip(flat, 1e-9, np.inf)
        slopes /= flat
        readvar /= flat**2
        poissonvar /= flat**2

    return slopes, readvar, poissonvar


def in_bounds(xx, yy, imbd, margin):
    """Filter sources to those landing on an image.

    Parameters
    ----------
    xx, yy: ndarray[nobj] (float)
        x & y positions of sources on image
    imbd : galsim.Image.Bounds
        bounds of image
    margin : int
        keep sources up to margin outside of bounds

    Returns
    -------
    keep : np.ndarray (bool)
        whether each source lands near the image (True) or not (False)
    """

    keep = ((xx > imbd.xmin - margin) & (xx < imbd.xmax + margin) & (
        yy > imbd.ymin - margin) & (yy < imbd.ymax + margin))
    return keep


def trim_objlist(objlist, image):
    """Trim a Table of objects down to those falling near an image.

    Objects must fall in a circle centered at the center of the image with
    radius 1.1 times the separation between the center and corner of the image.

    In contrast to in_bounds, this doesn't require the x and y coordinates of
    the sources, and just uses the ra/dec directly without needing to do the
    WCS transformation.

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
                         flux_to_counts_factor, bandpass=None, filter_name=None,
                         rng=None, seed=None):
    """Add sources to an image.

    Note: this includes Poisson noise when photon shooting is used
    (i.e., for chromatic source profiles), and otherwise is noise free.

    Parameters
    ----------
    image : galsim.Image
        Image to which to add sources with associated WCS.
    objlist : list[CatalogObject]
        Objects to add to image
    xpos, ypos : array_like
        x & y positions of sources (pixel) at which sources should be added
    psf : galsim.Profile
        PSF for image
    flux_to_counts_factor : float
        physical fluxes in objlist (whether in profile SEDs or flux arrays)
        should be multiplied by this factor to convert to total counts in the
        image
    bandpass : galsim.Bandpass
        bandpass in which image is being rendered.  This is used only in cases
        where chromatic profiles & PSFs are being used.
    filter_name : str
        filter to use to select appropriate flux from objlist.  This is only
        used when achromatic PSFs and sources are being rendered.
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    outinfo : np.ndarray
        Array structure containing rows for each source.  The columns give
        the total number of counts from the source entering the image and
        the time taken to render the source.
    """
    if rng is None and seed is None:
        seed = 143
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    log.info('Adding sources to image...')
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
        profile = obj.profile
        if not chromatic:
            if obj.flux is None:
                raise ValueError('Non-chromatic sources must have specified '
                                 'fluxes!')
            profile = profile.withFlux(obj.flux[filter_name])
        final = galsim.Convolve(profile * flux_to_counts_factor, psf)
        if chromatic:
            stamp = final.drawImage(
                bandpass, center=image_pos, wcs=image.wcs.local(image_pos),
                method='phot', rng=rng)
        else:
            try:
                stamp = final.drawImage(center=image_pos,
                                        wcs=image.wcs.local(image_pos))
            except galsim.GalSimFFTSizeError:
                log.warning(f'Skipping source {i} due to too '
                            f'large FFT needed for desired accuracy.')
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
                            filter_name=None, rng=None, seed=None):
    """Add some simulated counts to an image.

    No Roman specific code allowed!  To do this, we need to have an image
    to start with with an attached WCS.  We also need an exposure time
    and potentially a zpflux so we know how to translate between the catalog
    fluxes and the counts entering the image.  For chromatic rendering, this
    role instead is played by the bandpass, though the exposure time is still
    needed to handle that part of the conversion from flux to counts.

    Then there are a few of individual components that can be added on to
    an image:

    * objlist: a list of CatalogObjects to render, or a Table.  Can be chromatic
      or not.  This will have all your normal PSF and galaxy profiles.
    * sky: a sky background model.  This is different from a dark in that
      it is sensitive to the flat field.
    * dark: a dark model.
    * flat: a flat field for modulating the object and sky counts

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
        For non-chromatic profiles, the factor converting flux to counts / s.
    sky : float or array_like
        Image or constant with the counts / pix / sec from sky.
    dark : float or array_like
        Image or constant with the counts / pix / sec from dark current.
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
        image.array[:, :] = np.random.binomial(
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
                    darkrate=None, flat=None):
    """Simulate total counts in a single SCA.

    This gives the total counts in an idealized instrument with no systematics;
    it includes only distortion & PSF convolution.

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

    Returns
    -------
    image : galsim.Image
        idealized image of scene as seen by Roman, giving total electron counts
        from rate sources (astronomical objects; backgrounds; dark current) in
        each pixel.
    simcatobj : np.ndarray
        catalog of simulated objects in image
    """

    ma_table = parameters.ma_table[
        metadata['exposure']['ma_table_number']]

    sca = int(metadata['instrument']['detector'][3:])
    exptime = parameters.read_time * (ma_table[-1][0] + ma_table[-1][1] - 1)
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
                                 chromatic=chromatic, webbpsf=webbpsf)
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


def simulate(metadata, objlist,
             usecrds=True, webbpsf=True, level=2, crparam=dict(),
             persistence=None, seed=None, rng=None, **kwargs
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

    objlist : list[CatalogObject] or Table
        List of objects in the field to simulate
    usecrds : bool
        use CRDS to get distortion maps
    webbpsf : bool
        use webbpsf to generate PSF
    level : int
        0, 1 or 2, specifying level 1 or level 2 image
        0 makes a special idealized 'counts' image
    persistence : romanisim.persistence.Persistence
        persistence object to use; None for no persistence
    crparam : dict
        Parameters for cosmic ray simulations.  None for no cosmic rays.
        Empty dictionary for default parameters.
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.

    Returns
    -------
    image : roman_datamodels model
        simulated image
    extras : dict
        Dictionary of additionally valuable quantities.  Includes at least
        simcatobj, the image positions and fluxes of simulated objects.  It may
        also include information on persistence and cosmic ray hits.
    """

    meta = maker_utils.mk_common_meta()
    meta["photometry"] = maker_utils.mk_photometry()

    for key in parameters.default_parameters_dictionary.keys():
        meta[key].update(parameters.default_parameters_dictionary[key])

    util.add_more_metadata(meta)

    for key in metadata.keys():
        meta[key].update(metadata[key])

    # Create Image model to track validation
    image_node = maker_utils.mk_level2_image()
    image_node['meta'] = meta
    image_mod = roman_datamodels.datamodels.ImageModel(image_node)

    ma_table_number = image_mod.meta.exposure.ma_table_number
    filter_name = image_mod.meta.instrument.optical_element

    ma_table = parameters.ma_table[ma_table_number]
    exptime_tau = ((ma_table[-1][0] + (ma_table[-1][1] / 2))
                   * parameters.read_time)

    # TODO: replace this stanza with a function that looks at the metadata
    # and keywords and returns a dictionary with all of the relevant reference
    # data in numpy arrays.
    # should query CRDS for any reference files not specified on the command
    # line.
    reffiles = {}
    if usecrds:
        import crds
        refnames_lst = ['readnoise', 'dark', 'gain', 'inverselinearity', 'linearity', 'saturation']

        for key, value in parameters.reference_file_names.items():
            if (key in refnames_lst) and (value is not None):
                reffiles[key] = value
                refnames_lst.remove(key)

        if refnames_lst:
            reffiles.update(crds.getreferences(
                image_mod.get_crds_parameters(), reftypes=refnames_lst,
                observatory='roman'))

        read_noise_model = roman_datamodels.datamodels.ReadnoiseRefModel(
            reffiles['readnoise'])
        dark_model = roman_datamodels.datamodels.DarkRefModel(
            reffiles['dark'])
        gain_model = roman_datamodels.datamodels.GainRefModel(
            reffiles['gain'])
        inv_linearity_model = roman_datamodels.datamodels.InverselinearityRefModel(
            reffiles['inverselinearity'])
        linearity_model = roman_datamodels.datamodels.LinearityRefModel(
            reffiles['linearity'])
        saturation_model = roman_datamodels.datamodels.SaturationRefModel(
            reffiles['saturation'])

        try:
            flatfile = crds.getreferences(
                image_mod.get_crds_parameters(),
                reftypes=['flat'], observatory='roman')['flat']

            flat_model = roman_datamodels.datamodels.FlatRefModel(flatfile)
            flat = flat_model.data

        except crds.core.exceptions.CrdsLookupError:
            log.warning('Could not find flat; using 1')
            flat = 1

        # convert the last dark resultant into a dark rate by dividing by the
        # mean time in that resultant.
        nborder = parameters.nborder
        read_noise = read_noise_model.data[nborder:-nborder, nborder:-nborder]
        darkrate = dark_model.data[-1, nborder:-nborder, nborder:-nborder] / exptime_tau
        dark = dark_model.data[:, nborder:-nborder, nborder:-nborder]
        gain = gain_model.data[nborder:-nborder, nborder:-nborder]
        inv_linearity = nonlinearity.NL(
            inv_linearity_model.coeffs[:, nborder:-nborder, nborder:-nborder],
            inv_linearity_model.dq[nborder:-nborder, nborder:-nborder],
            gain=gain)
        linearity = nonlinearity.NL(
            linearity_model.coeffs[:, nborder:-nborder, nborder:-nborder],
            linearity_model.dq[nborder:-nborder, nborder:-nborder],
            gain=gain)
        darkrate *= gain
        image_mod.meta.ref_file.crds.sw_version = crds.__version__
        image_mod.meta.ref_file.crds.context_used = crds.get_context_name(
            observatory=image_mod.crds_observatory
        )
        saturation = saturation_model.data[nborder:-nborder, nborder:-nborder]
    else:
        read_noise = parameters.reference_file_names["readnoise"] \
            if parameters.reference_file_names["readnoise"] is not None else galsim.roman.read_noise
        darkrate = (parameters.reference_file_names["dark"] / exptime_tau) \
            if parameters.reference_file_names["dark"] is not None else galsim.roman.dark_current
        dark = parameters.reference_file_names["dark"]
        gain = parameters.reference_file_names["gain"]
        flat = parameters.reference_file_names["flat"] \
            if parameters.reference_file_names["flat"] is not None else 1
        inv_linearity = parameters.reference_file_names["inverselinearity"]
        linearity = parameters.reference_file_names["linearity"]
        saturation = parameters.reference_file_names["saturation"]

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
        webbpsf=webbpsf, flat=flat)
    if level == 0:
        im = dict(data=counts.array)
    else:
        l1, l1dq = romanisim.l1.make_l1(
            counts, ma_table_number, read_noise=read_noise, rng=rng, gain=gain,
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
        slopeinfo = make_l2(l1, ma_table, read_noise=read_noise,
                            gain=gain, flat=flat, linearity=linearity,
                            dark=dark, dq=l1dq)
        l2dq = np.bitwise_or.reduce(l1dq, axis=0)
        im, extras = make_asdf(
            *slopeinfo, metadata=image_mod.meta, persistence=persistence,
            dq=l2dq, imwcs=counts.wcs)
    else:
        extras = dict()

    if reffiles:
        extras["simulate_reffiles"] = {}
        for key, value in reffiles.items():
            extras["simulate_reffiles"][key] = value

    extras['simcatobj'] = simcatobj
    log.info('Simulation complete.')
    return im, extras


def make_test_catalog_and_images(
        seed=12345, sca=7, filters=None, nobj=1000,
        usecrds=True, webbpsf=True, galaxy_sample_file_name=None, **kwargs):
    """This routine kicks the tires on everything in this module."""
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
              filepath=None, persistence=None, dq=None, imwcs=None):
    """Wrap a galsim simulated image with ASDF/roman_datamodel metadata.

    Eventually this needs to get enough info to reconstruct a refit WCS.
    """

    out = maker_utils.mk_level2_image()
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
        out['meta'].update(metadata)

    if imwcs is not None:  # add a WCS
        if isinstance(imwcs, wcs.GWCS):
            out['meta'].update(wcs=imwcs.wcs)
        else:
            # make a gwcs WCS from a galsim.roman WCS
            out['meta'].update(
                wcs=wcs.wcs_from_fits_header(imwcs.header.header))

    out['data'] = slope
    out['dq'] = np.zeros(slope.shape, dtype='u4')
    if dq is not None:
        out['dq'][:, :] = dq
    out['var_poisson'] = slopevar_poisson
    out['var_rnoise'] = slopevar_rn
    out['var_flat'] = slopevar_rn * 0
    out['err'] = np.sqrt(out['var_poisson'] + out['var_rnoise'] + out['var_flat'])
    extras = dict()
    if persistence is not None:
        extras['persistence'] = persistence.to_dict()
    if filepath:
        af = asdf.AsdfFile()
        af.tree = {'roman': out, 'romanisim': extras}
        af.write_to(filepath)
    return out, extras
