"""Catalog generation and reading routines.

This module provides basic routines to allow romanisim to render scenes
based on catalogs of sources in those scenes.
"""
import dataclasses
import numpy as np
import galsim
from galsim import roman
from astropy import coordinates
from astropy import table
from astropy import units as u
from . import util
import romanisim.bandpass


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
                             seed=None,
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

    Returns
    -------
    astropy.table.Table
        Table including fields needed to generate a list of CatalogObject
        entries for rendering.
    """
    t1 = make_galaxies(coord, radius=radius, rng=rng, n=int(nobj * 0.8),
                       bandpasses=bandpasses)
    t2 = make_stars(coord, radius=radius, rng=rng, n=int(nobj * 0.1),
                    bandpasses=bandpasses)
    t3 = make_stars(coord, radius=radius / 100, rng=rng, n=int(nobj * 0.1),
                    bandpasses=bandpasses, truncation_radius=radius * 0.3)
    return table.vstack([t1, t2, t3])


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
        faintest AB magnitude for which to generate sources
        Note this magnitude is in a "fiducial" band which is not observed.
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


def table_to_catalog(table, bandpasses):
    """Read a astropy Table into a list of CatalogObjects.

    We want to read in a catalog and make a list of CatalogObjects.  The table
    must have the following columns:

    * ra : float, right ascension in degrees
    * dec : float, declination in degrees
    * type : str, 'PSF' or 'SER' for PSF or sersic profiles respectively
    * n : float, sersic index
    * half_light_radius : float, half light radius in arcsec
    * pa : float, position angle of ellipse relative to north (on the sky) in degrees
    * ba : float, ratio of semiminor axis b over semimajor axis a

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


def read_catalog(filename, bandpasses):
    """Read a catalog into a list of CatalogObjects.

    Catalog must be readable by astropy.table.Table.read(...) and contain
    columns enumerated in the docstring for table_to_catalog(...).

    Parameters
    ----------
    filename : str
        filename of catalog to read
    bandpasses : list[str]
        bandpasses for which fluxes are tabulated in the catalog

    Returns
    -------
    list[CatalogObject]
        list of catalog objects in filename
    """
    return table_to_catalog(table.Table.read(filename), bandpasses)
