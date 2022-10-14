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


@dataclasses.dataclass
class CatalogObject:
    """Simple class to hold galsim positions and profiles of objects."""
    sky_pos: galsim.CelestialCoord
    profile: galsim.GSObject
    flux : dict


def make_dummy_catalog(coord, radius=0.1, rng=None, seed=42, nobj=1000,
                       chromatic=True):
    """Make a dummy catalog for testing purposes.

    Parameters
    ----------
    coord : galsim.CelestialCoordinate
        radius around which to generate sources
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

    cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area,
                                exptime=1)
    cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area,
                                exptime=1)

    if chromatic:
        # following Roman demo13, all stars currently have the SED of Vega.
        # fluxes are set to have a specific value in the y bandpass.
        vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
        y_bandpass = roman.getBandpasses(AB_zeropoint=True)['Y106']

    objlist = []
    locs = util.random_points_in_cap(coord, radius, nobj, rng=rng)
    for i in range(nobj):
        sky_pos = locs[i]
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
            mu = np.log(mu_x**2 / (mu_x**2+sigma_x**2)**0.5)
            sigma = (np.log(1 + sigma_x**2/mu_x**2))**0.5
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


def make_dummy_table_catalog(coord, radius=0.1, rng=None, nobj=1000,
                             bandpasses=None):
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
    if bandpasses is None:
        bandpasses = roman.getBandpasses().keys()
    locs = util.random_points_in_cap(coord, radius, nobj, rng=rng)
    # n ~ 10^(3m/5) is what one gets for standard columns in a flat universe
    # cut off at 26th mag, go arbitrarily bright.
    # at least not crazy for a dummy catalog
    faintmag = 26 - 3  # get some brighter sources!
    hlr_at_faintmag = 0.6  # arcsec
    mag = faintmag - np.random.exponential(size=nobj, scale=5/3/np.log(10))
    # okay, now we need to mark some star/galaxy decisions.
    sersic_index = np.random.uniform(low=1, high=4.0, size=nobj)
    star = np.random.uniform(size=nobj) < 0.1
    sersic_index[star] = -1
    types = np.zeros(nobj, dtype='U3')
    types[:] = 'SER'
    types[star] = 'PSF'
    pa = np.random.uniform(size=nobj, low=0, high=360)
    pa[star] = 0
    # no clue what a realistic distribution of b/a is, but this at least goes to zero
    # for little tiny needles and peaks around circular objects, which isn't nuts.
    ba = np.random.beta(3, 1, size=nobj)
    ba = np.clip(ba, 0.2, 1)
    ba[star] = 1
    # ugh.  Half light radii should correlate with magnitude, with some scatter.
    hlr = 10**((faintmag - mag)/5) * hlr_at_faintmag
    # hlr is hlr_at_faintmag for faintmag sources
    # and let's put some log normal distribution on top of this
    hlr *= np.clip(np.exp(np.random.randn(nobj)*0.5), 0.1, 10)
    # let's not make anything too too small.
    hlr[hlr < 0.01] = 0.01
    hlr[star] = 0

    out = table.Table()
    out['ra'] = [x.ra.deg for x in locs]
    out['dec'] = [x.dec.deg for x in locs]
    out['type'] = types
    out['n'] = sersic_index
    out['half_light_radius'] = hlr
    out['pa'] = pa
    out['ba'] = ba
    for bandpass in bandpasses:
        mag_thisband = mag + np.random.randn(nobj)
        # sigma of one mag isn't nuts.  But this will be totally uncorrelated
        # in different bands, so we'll get some weird colored objects
        out[bandpass] = 10.**(-mag_thisband/2.5)
        # maggies!  what units should I actually pick here?
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
        pos = coordinates.SkyCoord(table['ra'][i]*u.deg, table['dec'][i]*u.deg,
                                   frame='icrs')
        pos = util.celestialcoord(pos)
        fluxes = {bp: table[bp][i] for bp in bandpasses}
        if table['type'][i] == 'PSF':
            obj = galsim.DeltaFunction()
        elif table['type'][i] == 'SER':
            obj = galsim.Sersic(table['n'][i], table['half_light_radius'][i])
            obj = obj.shear(
                q=table['ba'][i], beta=(table['pa'][i]+np.pi/2)*galsim.radians)
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
