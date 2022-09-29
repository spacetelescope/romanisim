"""Roman bandpass routines

The primary purpose of this module is to provide the number of counts
per second expected for sources observed by Roman given a source with
the nominal flat AB spectrum of 3631 Jy.  The ultimate source of this
information is https://roman.gsfc.nasa.gov/science/WFI_technical.html .
"""
import os
import pkg_resources
from scipy import integrate
from astropy import constants
from astropy.table import Table
from astropy import units as u

# to go from calibrated fluxes in maggies to counts in the Roman bands
# we need to multiply by a constant determined by the AB magnitude
# system and the shape of the Roman bands.
# The constant is \int (3631 Jy) (1/hv) T(v) dv
# T(v) should be the effective area at each wavelength, I guess
# divided by some nominal overall effective area.


# provide some translation dictionaries for the mapping from
# the galsim bandpass names to the Roman bandpass names and vice versa.
# it would be nice to be agnostic about which one we use.
galsim_bandpasses = ['Z087', 'Y106', 'J129', 'H158', 'F184', 'W149']
galsim2roman_bandpass = {x: 'F'+x[1:] for x in galsim_bandpasses}
roman2galsim_bandpass = {v: k for k, v in galsim2roman_bandpass.items()}

# provide some no-ops if we are given a key in the right bandpass
galsim2roman_bandpass.update(**{k: k for k in roman2galsim_bandpass})
roman2galsim_bandpass.update(**{k: k for k in galsim_bandpasses})


def read_gsfc_effarea(filename=None):
    """Read an effective area file from Roman.

    This just puts together the right invocation to get an Excel-converted
    CSV file into memory.

    Parameters
    ----------
    filename : str
        filename to read in

    Returns
    -------
    astropy.table.Table
        table with effective areas for different Roman bandpasses.
    """
    if filename is None:
        dirname = pkg_resources.resource_filename('romanisim', 'data')
        filename = os.path.join(dirname, 'Roman_effarea_20201130.csv')
    return Table.read(filename, format='csv', delimiter=',', header_start=1,
                      data_start=2)


def compute_abflux(effarea=None):
    """Compute the AB zero point fluxes for each filter.

    How many photons would a zeroth magnitude AB star deposit in
    Roman's detectors in a second?

    Parameters
    ----------
    effarea : astropy.Table.table
        Table from GSFC with effective areas for each filter.

    Returns
    -------
    dict[str] : float
        lookup table of zero point fluxes for each filter (photons / s)
    """

    if effarea is None:
        effarea = read_gsfc_effarea()

    # get the filter names.  This is a bit ugly since there's also
    # a wavelength column 'Wave', and Excel appends a column to each line
    # which astropy then gives a default name col12 to.
    filter_names = [x for x in effarea.dtype.names
                    if x != 'Wave' and 'col' not in x]
    abfv = 3631e-23 * u.erg / (u.s * u.cm**2 * u.hertz)
    out = dict()
    for bandpass in filter_names:
        integrand = abfv * constants.c / (
            effarea['Wave']*u.micron)**2  # f_lambda
        integrand /= constants.h * constants.c / (
            effarea['Wave']*u.micron)  # hc/lambda
        integrand *= effarea[bandpass]*u.m**2  # effective area in filter
        # integrate.simpson looks like it loses units.  So convert to something
        # we know about.
        integrand = integrand.to(1/(u.s*u.micron)).value
        zpflux = integrate.simpson(integrand, effarea['Wave'])
        # effarea['Wave'] is in microns, so we're left with a number of counts
        # per second
        out[bandpass] = zpflux
    return out


def get_abflux(bandpass):
    """Get the zero point flux for a particular bandpass.

    This is a simple wrapper for compute_abflux, caching the results.

    Parameters
    ----------
    bandpass : str
        the name of the bandpass

    Returns
    -------
    float
        the zero point flux (photons / s)s
    """
    bandpass = galsim2roman_bandpass.get(bandpass, bandpass)
    abflux = getattr(get_abflux, 'abflux', None)
    if abflux is None:
        abflux = compute_abflux()
        get_abflux.abflux = abflux
    return abflux[bandpass]
