import numpy as np
from astropy import table, coordinates, units as u


def gaia2romanisimcat(gaiacat, date, refepoch=2016.0, boost_parallax=1,
                      fluxfields=['F184']):
    """Convert Gaia output to a catalog of locations for input to romanisim.

    Mostly an astrometry routine, doing the usual parallax and proper motion
    computations by hand.  Caveat emptor---may all be wrong!

    boost_parallax is intended to allow accounting for the fact that parallaxes
    are for 1 AU, but Roman will be at 1.01 AU (L2) and so everything moves by
    1% more than you might expect.  If one wants to do better than that one
    needs to think about the detailed orbit of Roman in L2; the location
    of L2 itselfs moves annually and monthly due to the eccentricity of the
    Earth's orbit and the fact that the Earth is not at the Earth-moon
    barycenter.

    Parameters
    ----------
    gaiacat : astropy.table.Table
        Gaia catalog, containing at least ra, dec, pmra, pmdec, parallax,
        phot_g_mean_mag fields
    date : astropy.time.Time
        Time of observation
    refepoch : float
        Reference epoch of Gaia parameters
    boost_parallax : float
        Amount to boost parallaxes by (L2 describes a wider orbit than the
        Earth's)
    fluxfields : list[str]
        List of strings to fill with estimated fluxes

    Returns
    -------
    catalog : astropy.table.Table
        astropy Table formatted for input to romanisim
    """
    outcat = table.Table()
    dt = date.jyear - refepoch
    unitspherical = coordinates.UnitSphericalRepresentation(
        gaiacat['ra'], gaiacat['dec'])
    xyz = unitspherical.to_cartesian().xyz
    unit_vectors = unitspherical.unit_vectors()
    rahat = unit_vectors['lon'].xyz
    dechat = unit_vectors['lat'].xyz
    earthcoord = coordinates.get_body_barycentric('earth', date)
    earthcoord = earthcoord.xyz.to(u.AU).value
    radpermas = np.pi / (180 * 3600 * 1000)
    pmra = gaiacat['pmra'].to(u.mas / u.year).value
    pmdec = gaiacat['pmdec'].to(u.mas / u.year).value
    newxyz = (
        xyz + rahat * dt * radpermas * pmra + dechat * dt * radpermas * pmdec)
    plx = gaiacat['parallax'].to(u.mas).value * boost_parallax
    newxyz -= (rahat * earthcoord.dot(rahat) * plx * radpermas
               + dechat * earthcoord.dot(dechat) * plx * radpermas)
    # stars move in the opposite direction of the earth -> minus sign
    newunitspherical = coordinates.UnitSphericalRepresentation.from_cartesian(
        coordinates.CartesianRepresentation(newxyz))
    newra = newunitspherical.lon
    newdec = newunitspherical.lat
    outcat['ra'] = newra.to(u.deg).value
    outcat['dec'] = newdec.to(u.deg).value
    outcat['type'] = 'PSF'
    outcat['n'] = -1
    outcat['half_light_radius'] = -1
    outcat['pa'] = -1
    outcat['ba'] = -1
    for field in fluxfields:
        outcat[field] = 10. ** (-gaiacat['phot_g_mean_mag'] / 2.5)
    return outcat
