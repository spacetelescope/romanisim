"""Catalog generation and reading routines.

This module provides basic routines to allow romanisim to render scenes
based on catalogs of sources in those scenes.
"""
import os
import dataclasses
import numpy as np
import galsim
from galsim import roman
from astropy import coordinates, table
from astropy import units as u
from astropy.io import fits
import astropy_healpix
import astropy.time
from astroquery.gaia import Gaia
from romanisim import gaia as rsim_gaia
from . import util, log, parameters
import romanisim.bandpass

# COSMOS constants taken from the COSMOS2020 paper:
# https://arxiv.org/pdf/2110.13923
# Area of the ultra-deep regions of UltraVISTA data in square degrees
ULTRA_DEEP_AREA = 0.62

# COSMOS pixel scale
COSMOS_PIX_TO_ARCSEC = 0.15

# Filter interpolation coefficients
F146_J_COEFF = 0.46333417914234964
F158_H_COEFF = 0.823395077391525
F184_KS_COEFF = 0.3838145747397368

# Bandpass filters
BANDPASSES = set(romanisim.bandpass.galsim2roman_bandpass.values())


@dataclasses.dataclass
class CatalogObject:
    """Simple class to hold galsim positions and profiles of objects.

    Flux element contains the total AB flux from the source; i.e., the
    -2.5*log10(flux[filter_name]) would be the AB magnitude of the source.
    """
    sky_pos: galsim.CelestialCoord
    profile: galsim.GSObject
    flux: dict


def make_dummy_catalog(coord,
                       radius=0.1,
                       rng=None,
                       seed=42,
                       nobj=1000,
                       chromatic=True,
                       galaxy_sample_file_name=None,
                       ):
    """Make a dummy catalog for testing purposes.

    Parameters
    ----------
    coord : galsim.CelestialCoordinate
        center around which to generate sources
    radius : float
        radius (deg) within which to generate sources
    rng : Galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating random number generator.  Only used if rng is None.
    nobj : int
        Number of objects to simulate.
    chromatic : bool
        Use chromatic objects rather than gray objects.  The PSF of chromatic
        objects depends on their SED, while for gray objects this dependence is
        neglected.

    Returns
    -------
    list[CatalogObject]
        list of catalog objects to render
    """
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    if galaxy_sample_file_name is None:
        cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area,
                                    exptime=1)
        cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area,
                                    exptime=1)
    else:
        cat1 = galsim.COSMOSCatalog(galaxy_sample_file_name,
                                    area=roman.collecting_area, exptime=1)
        cat2 = cat1

    if chromatic:
        # following Roman demo13, all stars currently have the SED of Vega.
        # fluxes are set to have a specific value in the y bandpass.
        vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
        y_bandpass = roman.getBandpasses(AB_zeropoint=True)['Y106']

    objlist = []
    locs = util.random_points_in_cap(coord, radius, nobj, rng=rng)
    for i in range(nobj):
        sky_pos = util.celestialcoord(locs[i])
        p = rng()
        # prescription follows galsim demo13.
        if p < 0.8:  # 80% of targets; faint galaxies
            obj = cat1.makeGalaxy(chromatic=chromatic, gal_type='parametric',
                                  rng=rng)
            theta = rng() * 2 * np.pi * galsim.radians
            obj = obj.rotate(theta)
        elif p < 0.9:  # 10% of targets; stars
            mu_x = 1.e5
            sigma_x = 2.e5
            mu = np.log(mu_x**2 / (mu_x**2 + sigma_x**2)**0.5)
            sigma = (np.log(1 + sigma_x**2 / mu_x**2))**0.5
            gd = galsim.GaussianDeviate(rng, mean=mu, sigma=sigma)
            flux = np.exp(gd()) / roman.exptime
            if chromatic:
                sed = vega_sed.withFlux(flux, y_bandpass)
                obj = galsim.DeltaFunction() * sed
            else:
                obj = galsim.DeltaFunction().withFlux(flux)
        else:  # 10% of targets; bright galaxies
            obj = cat2.makeGalaxy(chromatic=chromatic, gal_type='parametric',
                                  rng=rng)
            obj = obj.dilate(2) * 4
            theta = rng() * 2 * np.pi * galsim.radians
            obj = obj.rotate(theta)
        objlist.append(CatalogObject(sky_pos, obj, None))
    return objlist


def make_dummy_table_catalog(coord,
                             radius=0.1,
                             rng=None,
                             nobj=1000,
                             bandpasses=None,
                             cosmos=False,
                             **kwargs
                             ):
    """Make a dummy table catalog.

    Fluxes are assigned to bands at random.  Locations are random within the
    spherical cap defined by coord and radius.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Location around which to generate catalog
    radius : float
        Radius in degrees of spherical cap in which to generate sources
    rng : galsim.BaseDeviate
        Random number generator to use
    nobj : int
        Number of objects to generate in spherical cap.
    bandpasses : list[str]
        List of names of bandpasses in which to generate fluxes.
    cosmos : Bool
        Flag to specify random selection of COSMOS galaxies

    Returns
    -------
    astropy.table.Table
        Table including fields needed to generate a list of CatalogObject
        entries for rendering.
    """
    # Create galaxies
    if cosmos:
        t1 = make_cosmos_galaxies(coord, radius=radius, rng=rng,
                                  bandpasses=bandpasses, **kwargs)
        t1 = t1[:int(nobj * 0.8)]
    else:
        t1 = make_galaxies(coord, radius=radius, rng=rng, n=int(nobj * 0.8),
                           bandpasses=bandpasses)

    # Create stars
    t2 = make_stars(coord, radius=radius, rng=rng, n=int(nobj * 0.1),
                    bandpasses=bandpasses)
    t3 = make_stars(coord, radius=radius / 100, rng=rng, n=int(nobj * 0.1),
                    bandpasses=bandpasses, truncation_radius=radius * 0.3)
    cat_table = table.vstack([t1, t2, t3])

    return cat_table


def make_cosmos_galaxies(coord,
                         radius=0.1,
                         bandpasses=None,
                         rng=None,
                         seed=50,
                         filename=None,
                         cat_area=None,
                         **kwargs
                         ):
    """Make a catalog of galaxies from sources in the COSMOS catalog.
    https://cosmos2020.calet.org/

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Location around which to generate sources.
    radius : float
        Radius in degrees in which to uniformly generate sources.
    bandpasses : list[str]
        List of names of bandpasses in which to generate fluxes.
    rng : galsim.BaseDeviate
        Random number generator to use.
    seed : int
        Seed to use for random numbers, only used if rng is None.
    filename : string
        Optional filename of a catalog of galaxies to draw from.
        Code assumes a format similar to that of COSMOS.
    cat_area : float
        Area of catalog file in square degrees.

    Returns
    -------
    catalog : astropy.Table
        Table for use with table_to_catalog to generate catalog for simulation.
    """
    from pathlib import Path

    if rng is None:
        rng = galsim.UniformDeviate(seed)

    # Generate list of required COSMOS filters (and Roman, if necessary)
    cos_filt = []
    if bandpasses is None:
        cos_filt = ['HSC_r_FLUX_AUTO', 'HSC_z_FLUX_AUTO', 'UVISTA_Y_FLUX_AUTO',
                    'UVISTA_J_FLUX_AUTO', 'UVISTA_H_FLUX_AUTO', 'UVISTA_Ks_FLUX_AUTO']
        bandpasses = BANDPASSES
    else:
        for opt_elem in bandpasses:
            if opt_elem == "F062":
                cos_filt.append('HSC_r_FLUX_AUTO')
            if opt_elem == "F087":
                cos_filt.append('HSC_z_FLUX_AUTO')
            if opt_elem == "F106":
                cos_filt.append('UVISTA_Y_FLUX_AUTO')
            if opt_elem in ("F129", "F158", "F146"):
                cos_filt.append('UVISTA_J_FLUX_AUTO')
            if opt_elem in ("F213", "F184"):
                cos_filt.append('UVISTA_Ks_FLUX_AUTO')
            if opt_elem in ("F158", "F184", "F146"):
                cos_filt.append('UVISTA_H_FLUX_AUTO')

    # Open COSMOS file and pare to required tabs
    if filename:
        cos_cat_all = table.Table.read(filename, format='fits', hdu=1)
    else:
        cos_cat_all = table.Table.read(Path(__file__).parent / "data" / "COSMOS2020_CLASSIC_R1_v2.2_p3_Streamlined.fits",
                                       format='fits', hdu=1)

    # Select galaxies
    cos_cat_all = cos_cat_all[(cos_cat_all['lp_type'] % 2) == 0]

    # Ensure that we are using only the ultra-deep regions of UltraVISTA data
    cos_cat_all = cos_cat_all[cos_cat_all['FLAG_UDEEP'] == 0]

    # Calculate source density
    if cat_area:
        cos_density = len(cos_cat_all['ID']) / cat_area
    else:
        cos_density = len(cos_cat_all['ID']) / ULTRA_DEEP_AREA

    # Calculate total sources
    sim_count = cos_density * np.pi * (radius * u.deg)**2

    # Only keep items with a flux radius
    cos_cat_all = cos_cat_all[cos_cat_all['FLUX_RADIUS'] > 0]

    # Only keep items with shape measurements
    cos_cat_all = cos_cat_all[cos_cat_all['ACS_B_WORLD'] > 0]
    cos_cat_all = cos_cat_all[cos_cat_all['ACS_A_WORLD'] > 0]

    # Set negative fluxes to zero
    for opt_elem in cos_filt:
        cos_cat_all[opt_elem][cos_cat_all[opt_elem] < 0]= 0

    # Drop sources with no flux in the requested bandpasses
    cos_cat_zero = np.lib.recfunctions.structured_to_unstructured(cos_cat_all[cos_filt].as_array())
    cos_cat_all = cos_cat_all[cos_cat_zero.max(axis=1) > 0]

    # Filter for flags
    cos_filt += ["ID", "FLUX_RADIUS", "ACS_A_WORLD", "ACS_B_WORLD"]
    cos_filt = list(set(cos_filt))

    # Trim catalog
    cos_cat = cos_cat_all[cos_filt]

    # Obtain random sources from the catalog
    rng_numpy_seed = rng.raw()
    rng_numpy = np.random.default_rng(rng_numpy_seed)
    sim_count = rng_numpy.poisson(sim_count.value)
    sim_ids = rng_numpy.integers(size=sim_count, low=0, high=len(cos_cat["ID"])).tolist()
    sim_cat = cos_cat[sim_ids]

    # Match cosmos filters to roman filters
    for opt_elem in bandpasses:
        if opt_elem == "F062":
            sim_cat['FLUX_F062'] = sim_cat['HSC_r_FLUX_AUTO']
        elif opt_elem == "F087":
            sim_cat['FLUX_F087'] = sim_cat['HSC_z_FLUX_AUTO']
        elif opt_elem == "F106":
            sim_cat['FLUX_F106'] = sim_cat['UVISTA_Y_FLUX_AUTO']
        elif opt_elem == "F129":
            sim_cat['FLUX_F129'] = sim_cat['UVISTA_J_FLUX_AUTO']
        elif opt_elem == "F146":
            sim_cat['FLUX_F146'] = (F146_J_COEFF * sim_cat['UVISTA_J_FLUX_AUTO']) + \
                ((1 - F146_J_COEFF) * sim_cat['UVISTA_H_FLUX_AUTO'])
        elif opt_elem == "F158":
            sim_cat['FLUX_F158'] = (F158_H_COEFF * sim_cat['UVISTA_H_FLUX_AUTO']) + \
                ((1 - F158_H_COEFF) * sim_cat['UVISTA_J_FLUX_AUTO'])
        elif opt_elem == "F184":
            sim_cat['FLUX_F184'] = (F184_KS_COEFF * sim_cat['UVISTA_Ks_FLUX_AUTO']) + \
                ((1 - F184_KS_COEFF) * sim_cat['UVISTA_H_FLUX_AUTO'])
        elif opt_elem == "F213":
            sim_cat['FLUX_F213'] = sim_cat['UVISTA_Ks_FLUX_AUTO']
        else:
            log.warning(f'Unknown filter {opt_elem} skipped in object catalog creation.')

    # Randomize positions of the sources
    locs = util.random_points_in_cap(coord, radius, len(sim_ids), rng=rng)

    # Set profile types
    types = np.zeros(len(sim_ids), dtype='U3')
    types[:] = 'SER'

    # Return Table with source parameters
    out = table.Table()
    out['ra'] = locs.ra.to(u.deg).value
    out['dec'] = locs.dec.to(u.deg).value
    out['type'] = types

    # Randomize concentrations
    out['n'] = rng_numpy.uniform(low=1.0, high=4.0, size=len(sim_ids))

    # Scale this from pixels to output half_light_radius unit (arcsec)
    out['half_light_radius'] = sim_cat['FLUX_RADIUS'].astype('f4') * COSMOS_PIX_TO_ARCSEC

    # Set random position angles
    pa = rng_numpy.uniform(size=len(sim_ids), low=0, high=360)
    out['pa'] = pa.astype('f4')

    # Save b / a shape ratio
    out['ba'] = (sim_cat['ACS_B_WORLD'] / sim_cat['ACS_A_WORLD']).astype('f4')

    # Perturb source fluxes by ~20%
    source_pert = np.ones(len(sim_ids))
    source_pert += ((0.2) * rng_numpy.normal(size=len(sim_ids)))

    # Convert fluxes to maggies by converting to Jankskys and normalizing for zero-point
    for bandpass in bandpasses:
        # Perturb sources fluxes by 5% per bandwidth
        band_source_pert = ((0.05) * rng_numpy.normal(size=len(sim_ids)))

        # Convert fluxes to maggies by converting to Jankskys, normalizing for zero-point, and applying perturbations
        out[bandpass] = sim_cat[f'FLUX_{bandpass}'].value * (1 + source_pert + band_source_pert) / (3631 * 10**6)

    # Return output table
    return out


def make_galaxies(coord,
                  n,
                  radius=0.1,
                  index=None,
                  faintmag=26,
                  hlr_at_faintmag=0.6,
                  bandpasses=None,
                  rng=None,
                  seed=50,
                  ):
    """Make a simple parametric catalog of galaxies.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Location around which to generate sources.
    n : int
        number of sources to generate
    radius : float
        radius in degrees of cap in which to uniformly generate sources
    index : int
        power law index of magnitudes
    faintmag : float
        Faintest AB magnitude for which to generate sources.

        Note this magnitude is in a "fiducial" band that is not observed.
        Actual requested bandpasses are equal to this fiducial band plus
        1 mag of Gaussian noise.
    hlr_at_faintmag : float
        typical half light radius at faintmag (arcsec)
    bandpasses : list[str]
        list of names of bandpasses for which to generate fluxes.
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed to use for random numbers, only used if rng is None

    Returns
    -------
    catalog : astropy.Table
        Table for use with table_to_catalog to generate catalog for simulation.
    """
    if bandpasses is None:
        bandpasses = roman.getBandpasses().keys()
        bandpasses = [romanisim.bandpass.galsim2roman_bandpass[b]
                      for b in bandpasses]
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    locs = util.random_points_in_cap(coord, radius, n, rng=rng)
    rng_numpy_seed = rng.raw()
    rng_numpy = np.random.default_rng(rng_numpy_seed)

    if index is None:
        index = 3 / 5
    distance_index = 5 * index - 1
    # i.e., for a dlog10n / dm = 3/5 (uniform), corresponds to a
    # distance index of 2, which is the normal volume element
    # dn ~ rho * r^2 dr
    distance = rng_numpy.power(distance_index + 1, size=n)
    mag = faintmag + 5 * np.log10(distance)
    sersic_index = rng_numpy.uniform(low=1, high=4.0, size=n)
    types = np.zeros(n, dtype='U3')
    types[:] = 'SER'
    pa = rng_numpy.uniform(size=n, low=0, high=360)
    # no clue what a realistic distribution of b/a is, but this at least goes to zero
    # for little tiny needles and peaks around circular objects, which isn't nuts.
    ba = rng_numpy.beta(3, 1, size=n)
    ba = np.clip(ba, 0.2, 1)
    # Half light radii should correlate with magnitude, with some scatter.
    hlr = 10**((faintmag - mag) / 5) * hlr_at_faintmag
    # hlr is hlr_at_faintmag for faintmag sources
    # and let's put some log normal distribution on top of this
    hlr *= np.clip(np.exp(rng_numpy.normal(size=n) * 0.5), 0.1, 10)
    # let's not make anything too too small.
    hlr[hlr < 0.01] = 0.01

    out = table.Table()
    out['ra'] = locs.ra.to(u.deg).value
    out['dec'] = locs.dec.to(u.deg).value
    out['type'] = types
    out['n'] = sersic_index
    out['half_light_radius'] = hlr.astype('f4')
    out['pa'] = pa.astype('f4')
    out['ba'] = ba.astype('f4')
    for bandpass in bandpasses:
        mag_thisband = mag + rng_numpy.normal(size=n)
        # sigma of one mag isn't nuts.  But this will be totally uncorrelated
        # in different bands, so we'll get some weird colored objects
        out[bandpass] = (10.**(-mag_thisband / 2.5)).astype('f4')

    return out


def make_gaia_stars(coord,
                    radius=0.1,
                    date=None,
                    bandpasses=None,
                    **kwargs
                    ):
    """Make a catalog of stars from the Gaia catalog.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Location around which to generate sources.
    radius : float
        Radius in degrees in which to generate sources
    date : astropy.time.Time
        Optional argument to provide a date and time for stellar search
    bandpasses : list[str]
        List of names of bandpasses for which to generate fluxes.

    Returns
    -------
    catalog : astropy.Table
        Table for use with table_to_catalog to generate catalog for simulation.
    """

    if bandpasses is None:
        bandpasses = BANDPASSES

    if date is None:
        date = astropy.time.Time('2026-01-01T00:00:00')

    # Perform Gaia search
    q = f'select * from gaiadr3.gaia_source where distance({coord.ra.value}, {coord.dec.value}, ra, dec) < {radius}'
    job = Gaia.launch_job_async(q)
    r = job.get_results()

    # Create catalog
    star_cat = rsim_gaia.gaia2romanisimcat(r, date, fluxfields=bandpasses)

    return star_cat


def read_one_healpix(filename,
                     date=None,
                     bandpasses=None,
                     **kwargs
                     ):
    """Make a catalog of stars from a Gaia catalog files, sorted by Healpix.

    The files are assumed to be in FITS format.
    Healpix parameters:
    128 sides
    nested order
    Galactic frame

    Parameters
    ----------
    filename: string
        Path to healpix file
    date : astropy.time.Time
        Optional argument to provide a date and time for stellar search
    bandpasses : list[str]
        List of names of bandpasses for which to generate fluxes.

    Returns
    -------
    catalog : astropy.Table
        Table for use with table_to_catalog to generate catalog for simulation.
    """

    # Open healpix file
    cat_table = table.Table.read(filename)

    # Check for RSIM Gaia catalog
    if 'phot_g_mean_mag' in cat_table.colnames:
        return rsim_gaia.gaia2romanisimcat(cat_table, date, fluxfields=bandpasses, **kwargs)
    else:
        return cat_table


def make_stars(coord,
               n,
               radius=0.1,
               index=None,
               faintmag=26,
               truncation_radius=None,
               bandpasses=None,
               rng=None,
               seed=51,
               ):
    """Make a simple parametric catalog of stars.

    If truncation radius is None, this makes a uniform distribution.  If the
    truncation_radius is not None, it makes a King distribution where the
    core radius is given by the radius and the truncation radius is given by
    truncation_radius.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Location around which to generate sources.
    n : int
        number of sources to generate
    radius : float
        radius in degrees of cap in which to generate sources
    index : int
        power law index of magnitudes; uniform density & standard candle
        implies 3/5.
    faintmag : float
        faintest AB magnitude for which to generate sources

        Note this magnitude is in a "fiducial" band which is not observed.
        Actual requested bandpasses are equal to this fiducial band plus
        1 mag of Gaussian noise.
    truncation_radius : float
        truncation radius of cluster if not None; otherwise ignored.
    bandpasses : list[str]
        list of names of bandpasses for which to generate fluxes.
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed for random number generator to use, only used if rng is None

    Returns
    -------
    catalog : astropy.Table
        Table for use with table_to_catalog to generate catalog for simulation.
    """
    if bandpasses is None:
        bandpasses = roman.getBandpasses().keys()
        bandpasses = [romanisim.bandpass.galsim2roman_bandpass[b]
                      for b in bandpasses]
    if rng is None:
        rng = galsim.UniformDeviate(seed)
    rng_numpy_seed = rng.raw()
    rng_numpy = np.random.default_rng(rng_numpy_seed)

    if truncation_radius is None:
        locs = util.random_points_in_cap(coord, radius, n, rng=rng)
    else:
        locs = util.random_points_in_king(coord, radius, truncation_radius,
                                          n, rng=rng)
    if index is None:
        index = 3 / 5
    distance_index = 5 * index - 1
    # i.e., for a dlog10n / dm = 3/5 (uniform), corresponds to a
    # distance index of 2, which is the normal volume element
    # dn ~ rho * r^2 dr
    distance = rng_numpy.power(distance_index + 1, size=n)
    mag = faintmag + 5 * np.log10(distance)
    types = np.zeros(n, dtype='U3')
    types[:] = 'PSF'
    pa = mag * 0
    ba = mag * 0 + 1
    hlr = mag * 0
    sersic_index = mag * 0 - 1

    out = table.Table()
    out['ra'] = locs.ra.to(u.deg).value
    out['dec'] = locs.dec.to(u.deg).value
    out['type'] = types
    out['n'] = sersic_index.astype('f4')
    out['half_light_radius'] = hlr.astype('f4')
    out['pa'] = pa.astype('f4')
    out['ba'] = ba.astype('f4')
    for bandpass in bandpasses:
        mag_thisband = mag + rng_numpy.normal(size=n)
        # sigma of one mag isn't nuts.  But this will be totally uncorrelated
        # in different bands, so we'll get some weird colored objects
        out[bandpass] = (10.**(-mag_thisband / 2.5)).astype('f4')
    return out


def image_table_to_catalog(table, bandpasses):
    """Read an astropy Table into a list of CatalogObjects.

    We want to read in an image catalog and make a list of CatalogObjects.
    The image catalog indicates that specific galaxies stored as images in a
    RealGalaxyCatalog should be rendered at specific locations in the images,
    for example, if one had postage stamps of galaxies from a hydrodynamical
    simulation and wanted to render them in a Roman simulation.  The table
    must have the following columns:

    * ra : float, right ascension in degrees
    * dec : float, declination in degrees
    * ident: int, identity entry in RealGalaxyCatalog
    * dilate: float, how much to dilate the image by, in degrees
    * rotate: float, how much to rotate the image by
    * shear_ba: how much to shear the image by
                (new minor over major axis, if the original image were round)
    * shear_pa: angle to shear the image on, in degrees

    Additionally there must be a column for each bandpass giving the total flux
    in that bandbass, integrating over the image.

    The file name for the RealGalaxyCatalog must be present in the
    'real_galaxy_catalog_filename' keyword in the table metadata.

    Note that GalSim tries to deconvolve the image by the RealGalaxyCatalog
    PSF before reconvolving it with the appropriate filter PSF.  Depending
    on the dilation factor and the RealGalaxyCatalog PSF, this can require
    substantial deconvolution and lead to ringing.  For sufficiently large
    dilations, any initial PSF will become larger than the Roman PSF
    and induce ringing.

    Parameters
    ----------
    table : astropy.table.Table
        astropy Table containing ra, dec, ident, dilate, rotate, shear_amount,
        shear_pa, and fluxes in different bandpasses.
        Metadata must include real_galaxy_catalog_filename pointing to the
        RealGalaxyCatalog to use.
    bandpasses : list[str]
        list of names of bandpasses.  These bandpasses must have
        columns of the corresponding names in the catalog, containing
        the objects' fluxes.

    Returns
    -------
    list[CatalogObject]
        list of catalog objects for catalog
    """
    out = list()
    if 'real_galaxy_catalog_filename' not in table.meta:
        raise ValueError(
            'catalog file name must be present in table metadata.')
    rgc = galsim.RealGalaxyCatalog(table.meta['real_galaxy_catalog_filename'])
    for i in range(len(table)):
        pos = coordinates.SkyCoord(table['ra'][i] * u.deg, table['dec'][i] * u.deg,
                                   frame='icrs')
        pos = util.celestialcoord(pos)
        fluxes = {bp: table[bp][i] for bp in bandpasses}
        obj = galsim.RealGalaxy(rgc, id=table['ident'][i])
        obj = obj.shear(
            q=table['shear_ba'][i],
            beta=(np.radians(table['shear_pa'][i])
                  + np.pi / 2) * galsim.radians)
        obj = obj.dilate(table['dilate'][i])
        obj = obj.rotate(np.radians(table['rotate'][i]) * galsim.radians)
        out.append(CatalogObject(pos, obj, fluxes))
    return out


def make_image_catalog(image_filenames, psf, out_base_filename,
                       pixel_scale=0.05):
    """Construct a RealGalaxyCatalog from a list of image filenames.

    GalSim supports catalogs of real galaxies from input images that can be
    inserted into output images.  This function makes it easy to produce a
    set of files that can be used as a GalSim RealGalaxyCatalog from a list of
    fits input images.  These input images can come from anywhere, but are
    expected to come from either real imaging or from hydrodynamical simulations.

    This routine assumes that all images share a common PSF, which is given as the
    PSF argument.  This PSF is deconvolved before reconvolving with the
    appropriate filter-specific PSF when rendering a Roman image.

    Files are written to the provided base filename, plus
    ".fits", "_image.fits", and "_psf.fits" extensions.  The first file
    contains a binary table with some metadata about the included images;
    the second contains a file with one HDU for each image; the third includes
    the PSF image.

    Parameters
    ----------
    image_filenames : list[str]
        filenames of images of images to use
    psf : np.ndarray[float]
        image of PSF through which images were seen
    out_base_filename : str
        output filename to use for RealGalaxyCatalog files output here
    pixel_scale : float
        pixel scale of PSF and images
    """
    outdtype = [('ident', 'i4'), ('gal_filename', 'U200'), ('psf_filename', 'U200'),
                ('noise_file_name', 'U200'), ('gal_hdu', 'i4'), ('psf_hdu', 'i4'),
                ('pixel_scale', 'f4'), ('noise_variance', 'f4'), ('mag', 'f4'),
                ('band', 'U10'), ('weight', 'f4'), ('stamp_flux', 'f4')]
    nimage = len(image_filenames)
    res = np.zeros(nimage, dtype=outdtype)
    res['ident'] = np.arange(nimage)
    out_filename_image = out_base_filename + '_img.fits'
    hdul = fits.HDUList()
    for fn in image_filenames:
        hdu = fits.open(fn)[0]
        hdul.append(hdu)
    hdul.writeto(out_filename_image)
    res['gal_filename'] = out_filename_image
    res['gal_hdu'] = res['ident']
    out_filename_psf = (out_base_filename + '_psf.fits')
    fits.writeto(out_filename_psf, psf)
    res['psf_filename'] = out_filename_psf
    res['psf_hdu'] = 0
    res['pixel_scale'] = pixel_scale
    res['noise_variance'] = 0
    res['mag'] = 0  # not sure what this is used for
    res['stamp_flux'] = 1  # not sure what this is used for
    res['weight'] = 1 / nimage
    res['band'] = 'F087'
    fits.writeto(out_base_filename + '.fits', res)


def table_to_catalog(table, bandpasses):
    """Read an astropy Table into a list of CatalogObjects.

    We want to read in a catalog and make a list of CatalogObjects.  The table
    must have the following columns:

    * ra : float, right ascension in degrees
    * dec : float, declination in degrees
    * type : str, 'PSF' or 'SER' for PSF or sersic profiles respectively
    * n : float, sersic index
    * half_light_radius : float, half light radius in arcsec
    * pa : float, position angle of ellipse relative to north (on the sky) in degrees
    * ba : float, semiminor axis b divided by semimajor axis a

    Alternatively, the table must have the columns specified in
    image_table_to_catalog (for image input).

    Additionally there must be a column for each bandpass giving the flux
    in that bandbass.

    Parameters
    ----------
    table : astropy.table.Table
        astropy Table containing ra, dec, type, n, half_light_radius, pa, ba
        and fluxes in different bandpasses
    bandpasses : list[str]
        list of names of bandpasses.  These bandpasses must have
        columns of the corresponding names in the catalog, containing
        the objects' fluxes.

    Returns
    -------
    list[CatalogObject]
        list of catalog objects for catalog
    """

    if 'ident' in table.dtype.names:
        return image_table_to_catalog(table, bandpasses)

    out = list()
    for i in range(len(table)):
        pos = coordinates.SkyCoord(table['ra'][i] * u.deg, table['dec'][i] * u.deg,
                                   frame='icrs')
        pos = util.celestialcoord(pos)
        fluxes = {bp: table[bp][i] for bp in bandpasses}
        if table['type'][i] == 'PSF':
            obj = galsim.DeltaFunction()
        elif table['type'][i] == 'SER':
            obj = galsim.Sersic(table['n'][i], table['half_light_radius'][i])
            obj = obj.shear(
                q=table['ba'][i],
                beta=(np.radians(table['pa'][i]) + np.pi / 2) * galsim.radians)
        else:
            raise ValueError('Catalog types must be either PSF or SER.')
        out.append(CatalogObject(pos, obj, fluxes))
    return out


def read_catalog(filename,
                 coord,
                 date=None,
                 bandpasses=None,
                 radius=parameters.WFS_FOV,
                 **kwargs):
    """Read a catalog (or directory of catalogs) into a list of CatalogObjects.

    Catalog must be readable by astropy.table.Table.read(...) and contain
    columns enumerated in the docstring for table_to_catalog(...).

    Parameters
    ----------
    filename : str or None
        Filename of catalog or directory to read
        If None, will call Gaia website
    coord : astropy.coordinates.SkyCoord
        Location around which to generate sources.
    date : astropy.time.Time
        Optional argument to provide a date and time for stellar search
    bandpasses : list[str]
        Bandpasses for which fluxes are tabulated in the catalog
    radius: float
        Radius over which to search healpix for source file indicies

    Returns
    -------
    cat : astropy.Table
        Table for use with table_to_catalog to generate catalog for simulation.
    """
    # Set defaults if needed
    if bandpasses is None:
        bandpasses = BANDPASSES

    if date is None:
        date = astropy.time.Time('2026-01-01T00:00:00')

    # Generate star catalogs
    if filename is None:
        # Call Gaia website for information
        cat = make_gaia_stars(coord, radius=radius, date=date, **kwargs)
    elif os.path.isdir(filename):
        # Healpix catalogs within a directory

        # Set parameters of Healpix
        hp = astropy_healpix.HEALPix(nside=128, order='nested', frame=coordinates.Galactic())

        # Find Healpix
        hp_cone = hp.cone_search_skycoord(util.skycoord(coord), radius=radius * u.deg)

        # Create initial catalog
        hp_filename = filename + f"/cat-{hp_cone[0]}.fits"
        cat = read_one_healpix(hp_filename, date, bandpasses, **kwargs)

        # Append additional healpix catalogs
        if len(hp_cone > 1):
            for hp_idx in hp_cone[1:]:
                hp_filename = filename + f"/cat-{hp_idx}.fits"
                hp_table = read_one_healpix(hp_filename, date, bandpasses, **kwargs)
                cat = table.vstack([cat, hp_table])
    else:
        # Catalog file
        cat = table.Table.read(filename)

    # Remove bad entries
    bandpass = [f for f in cat.dtype.names if f in BANDPASSES]
    bad = np.zeros(len(cat), dtype='bool')
    for b in bandpass:
        bad |= ~np.isfinite(cat[b])
        if hasattr(cat[b], 'mask'):
            bad |= cat[b].mask
    cat = cat[~bad]
    nbad = np.sum(bad)
    if nbad > 0:
        log.info(f'Removing {nbad} catalog entries with non-finite or '
                 'masked fluxes.')

    return cat
