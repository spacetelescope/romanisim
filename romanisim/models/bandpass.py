import importlib.resources
import os

import numpy as np
from astropy import constants
from astropy import units as u
from astropy.io import ascii
from galsim import Bandpass, LookupTable
from galsim.errors import galsim_warn
from scipy import integrate

from .parameters import (
    collecting_area,
    roman_tech_repo_path,
)

effarea_root = os.path.join(
    roman_tech_repo_path, "data/WideFieldInstrument/Imaging/EffectiveAreas/"
)

data_root = str(importlib.resources.files("romanisim").joinpath("data/"))

# Which bands should use the long vs short pupil plane files for the PSF.
# F184, K213
longwave_bands = ["F184", "K213"]
# R062, Z087, Y106, J129, H158, W146, SNPrism, Grism_0thOrder, Grism_1stOrder.
# Note that the last three are not imaging bands.
non_imaging_bands = ["Grism_0thOrder", "Grism_1stOrder", "SNPrism"]
shortwave_bands = [
    "R062",
    "Z087",
    "Y106",
    "J129",
    "H158",
    "W146",
] + non_imaging_bands

# provide some translation dictionaries for the mapping from
# the galsim bandpass names to the Roman bandpass names and vice versa.
# it would be nice to be agnostic about which one we use.
galsim_bandpasses = [
    "R062",
    "Z087",
    "Y106",
    "J129",
    "H158",
    "F184",
    "K213",
    "W146",
]
galsim2roman_bandpass = {x: "F" + x[1:] for x in galsim_bandpasses}
roman2galsim_bandpass = {v: k for k, v in galsim2roman_bandpass.items()}

# provide some no-ops if we are given a key in the right bandpass
galsim2roman_bandpass.update(**{k: k for k in roman2galsim_bandpass})
roman2galsim_bandpass.update(**{k: k for k in galsim_bandpasses})

# to go from calibrated fluxes in maggies to counts in the Roman bands
# we need to multiply by a constant determined by the AB magnitude
# system and the shape of the Roman bands.
# The constant is \int (3631 Jy) (1/hv) T(v) dv
# T(v) should be the effective area at each wavelength, I guess
# divided by some nominal overall effective area.

# AB Zero Spectral Flux Density
ABZeroSpFluxDens = 3631e-23 * u.erg / (u.s * u.cm**2 * u.hertz)


def get_zodi_bkgnd(ecl_lat, ecl_dlon, lambda_min, lambda_max, Tlambda, T):
    # Computes the zodiacal foreground radiation in units of photons/m2/s/arcsec^2.

    # Inputs:
    #   ecl_lat = ecliptic latitude (input in *degrees*)
    #   ecl_dlon = ecliptic longitude relative to Sun (input in *degrees*)
    #   lambda_min = min wavelength (input in *microns*)
    #   lambda_max = max wavlenegth (input in *microns*)
    #   T = throughput table (NULL for unit throughput)

    # Caveats:
    #   No allowance is made for annual variations (due to tilt of Ecliptic relative to dust midplane)
    #   Range of valid wavelengths = 0.22--2.50 microns (in particlar: neglected thermal emission)
    deg = np.pi / 180.0  # /* degrees */
    Nlambda = 100  # /* number of integration points in wavelength */
    # /* Sky brightness table: rows are varying ecliptic latitude, cols are varying longitude,
    #  * at the values shown.
    #  *
    #  * This is at 0.5um in units of 1e-8 W/m^2/sr/um.
    #  * Electronic version of Table 17 of Leinert (1997), except for placeholders (1's) at
    #  * elongation <15 degrees (where we should not use this routine anyway!).
    #  */

    # fmt: off
    nlat = 11
    nlon = 19
    betaTable = [0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90]
    dlonTable = [0, 5, 10, 15, 20, 25, 30, 35, 40,
                 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    skyTable = [
        1,    1,    1, 3140, 1610, 985, 640, 275, 150, 100, 77,
        1,    1,    1, 2940, 1540, 945, 625, 271, 150, 100, 77,
        1,    1, 4740, 2470, 1370, 865, 590, 264, 148, 100, 77,
        11500, 6780, 3440, 1860, 1110, 755, 525, 251, 146, 100, 77,
        6400, 4480, 2410, 1410,  910, 635, 454, 237, 141,  99, 77,
        3840, 2830, 1730, 1100,  749, 545, 410, 223, 136,  97, 77,
        2480, 1870, 1220,  845,  615, 467, 365, 207, 131,  95, 77,
        1650, 1270,  910,  680,  510, 397, 320, 193, 125,  93, 77,
        1180,  940,  700,  530,  416, 338, 282, 179, 120,  92, 77,
        910,  730,  555,  442,  356, 292, 250, 166, 116,  90, 77,
        505,  442,  352,  292,  243, 209, 183, 134, 104,  86, 77,
        338,  317,  269,  227,  196, 172, 151, 116,  93,  82, 77,
        259,  251,  225,  193,  166, 147, 132, 104,  86,  79, 77,
        212,  210,  197,  170,  150, 133, 119,  96,  82,  77, 77,
        188,  186,  177,  154,  138, 125, 113,  90,  77,  74, 77,
        179,  178,  166,  147,  134, 122, 110,  90,  77,  73, 77,
        179,  178,  165,  148,  137, 127, 116,  96,  79,  72, 77,
        196,  192,  179,  165,  151, 141, 131, 104,  82,  72, 77,
        230,  212,  195,  178,  163, 148, 134, 105,  83,  72, 77
    ]
    # /* Solar spectrum: in units of W/m^2/sr/um at log10(lambda/um) = -0.80(0.01)+0.40
    #  * Ref: Colina, Bohlin, Castelli 1996 AJ 112, 307
    #  *
    #  * V band (550 nm) is SolarSpec[54]
    #  */
    SolarSpec = [
        1.87138e-01, 2.61360e-01, 4.08020e-01, 6.22197e-01, 9.02552e-01, 1.51036e+00, 2.25890e+00, 2.75901e+00, 4.03384e+00, 5.42817e+00,
        7.26182e+00, 1.01910e+01, 2.01114e+01, 3.62121e+01, 4.31893e+01, 5.43904e+01, 4.91581e+01, 4.95091e+01, 4.95980e+01, 5.93722e+01,
        5.27380e+01, 1.02502e+02, 1.62682e+02, 2.53618e+02, 2.01084e+02, 2.08273e+02, 4.05163e+02, 5.39830e+02, 5.31917e+02, 6.31200e+02,
        7.06134e+02, 8.13653e+02, 1.00508e+03, 9.56536e+02, 9.50568e+02, 9.82400e+02, 1.06093e+03, 1.12669e+03, 1.09922e+03, 1.10224e+03,
        1.36831e+03, 1.72189e+03, 1.74884e+03, 1.59871e+03, 1.74414e+03, 1.98823e+03, 2.02743e+03, 2.00367e+03, 2.03584e+03, 1.90296e+03,
        1.93097e+03, 1.86594e+03, 1.86655e+03, 1.87957e+03, 1.87978e+03, 1.83915e+03, 1.84447e+03, 1.80371e+03, 1.76779e+03, 1.70796e+03,
        1.66589e+03, 1.61456e+03, 1.53581e+03, 1.51269e+03, 1.44957e+03, 1.39215e+03, 1.34031e+03, 1.28981e+03, 1.24501e+03, 1.19548e+03,
        1.15483e+03, 1.10546e+03, 1.06171e+03, 9.94579e+02, 9.54006e+02, 9.15287e+02, 8.63891e+02, 8.31183e+02, 7.95761e+02, 7.62568e+02,
        7.27589e+02, 6.94643e+02, 6.60883e+02, 6.21830e+02, 5.83846e+02, 5.59624e+02, 5.34124e+02, 5.06171e+02, 4.80985e+02, 4.63139e+02,
        4.39482e+02, 4.13122e+02, 3.94543e+02, 3.75591e+02, 3.56069e+02, 3.35294e+02, 3.16374e+02, 2.98712e+02, 2.82737e+02, 2.69581e+02,
        2.49433e+02, 2.36936e+02, 2.21403e+02, 2.04770e+02, 1.87379e+02, 1.75880e+02, 1.60408e+02, 1.46210e+02, 1.36438e+02, 1.24412e+02,
        1.16500e+02, 1.07324e+02, 9.89669e+01, 9.12134e+01, 8.28880e+01, 7.71064e+01, 7.06245e+01, 6.42367e+01, 5.87697e+01, 5.39387e+01,
        4.98208e+01
    ]
    # fmt: on

    # /* Put longitude between 0 and 180 */
    ecl_dlon = np.abs(ecl_dlon)
    ecl_dlon -= 360 * np.floor(ecl_dlon / 360)
    if ecl_dlon > 180:
        ecl_dlon = 360 - ecl_dlon
    if ecl_dlon > 180:
        ecl_dlon = 180
    # /* Set latitude to be positive */
    ecl_lat = np.abs(ecl_lat)
    if ecl_lat > 90:
        ecl_lat = 90
    # /* Check wavelength ranges */
    # if (lambda_min<0.22 | lambda_max>2.50):
    #   print("Error: range lambda = %12.5lE .. %12.5lE microns out of range.\n", lambda_min, lambda_max);
    # /* Compute elongation (Sun angle). Complain if <15 degrees. */
    z = np.cos(ecl_lat * deg) * np.cos(ecl_dlon * deg)
    if z >= 1:
        elon = 0
    elif z <= -1:
        elon = 180
    else:
        elon = np.arccos(z) / deg
    if elon < 15:
        # print("Error: get_zodi_bkgnd: elongation = " +
        #       str(elon)+" degrees out of valid range.\n")
        return -10
    # /* Compute sky brightness at 0.5um in units of 1e-8 W/m^2/sr/um.
    #  * Fit to Table 17 of Leinert (1997).
    #  */
    ilat = 0
    while betaTable[ilat + 1] < ecl_lat and ilat < nlat - 2:
        ilat += 1
    ilon = 0
    while dlonTable[ilon + 1] < ecl_dlon and ilon < nlon - 2:
        ilon += 1
    frlat = (ecl_lat - betaTable[ilat]) / (betaTable[ilat + 1] - betaTable[ilat])
    frlon = (ecl_dlon - dlonTable[ilon]) / (dlonTable[ilon + 1] - dlonTable[ilon])
    sky05 = np.exp(
        np.log(skyTable[ilat + (ilon) * nlat]) * (1.0 - frlat) * (1.0 - frlon)
        + np.log(skyTable[ilat + (ilon + 1) * nlat]) * (1.0 - frlat) * (frlon)
        + np.log(skyTable[ilat + 1 + (ilon) * nlat]) * (frlat) * (1.0 - frlon)
        + np.log(skyTable[ilat + 1 + (ilon + 1) * nlat]) * (frlat) * (frlon)
    )
    # /* Integrate over wavelengths */
    zodi_tot = 0.0
    dlambda = (lambda_max - lambda_min) / float(Nlambda)
    for ilambda in range(Nlambda):
        lambda_ = lambda_min + (ilambda + 0.5) / Nlambda * (lambda_max - lambda_min)
        # /* Solar spectrum at this wavelength: F_lambda/F_{0.5um} */
        index_lambda = 100 * np.log(lambda_) / np.log(10.0) + 80
        ilam = int(np.floor(index_lambda))
        frlam = index_lambda - ilam
        sun_spec = (
            SolarSpec[ilam] + frlam * (SolarSpec[ilam + 1] - SolarSpec[ilam])
        ) / SolarSpec[50]
        # /* Color correction relative to solar */
        if lambda_ > 0.5:
            if elon > 90:
                fco = 1.0 + (0.6) * np.log(lambda_ / 0.5) / np.log(10.0)
            elif elon < 30:
                fco = 1.0 + (0.8) * np.log(lambda_ / 0.5) / np.log(10.0)
            else:
                fco = 1.0 + (0.8 - 0.2 * (elon - 30) / 60.0) * np.log(
                    lambda_ / 0.5
                ) / np.log(10.0)
        else:
            if elon > 90:
                fco = 1.0 + (0.9) * np.log(lambda_ / 0.5) / np.log(10.0)
            elif elon < 30:
                fco = 1.0 + (1.2) * np.log(lambda_ / 0.5) / np.log(10.0)
            else:
                fco = 1.0 + (1.2 - 0.3 * (elon - 30) / 60.0) * np.log(
                    lambda_ / 0.5
                ) / np.log(10.0)
        # /* The integral for the zodiacal foreground.
        #  * Here sky05*fco*sun_spec*dlambda is the power per unit area per unit solid angle in this
        #  * wavelength range (Units: 1e-8 W/m^2/sr).
        #  *
        #  * The conversion from 1e-8 W --> photons/sec is 5.03411747e10*lambda(um).
        #  */
        zodi_tot += (
            sky05
            * fco
            * sun_spec
            * dlambda
            * 5.03411747e10
            * lambda_
            * np.interp(lambda_, Tlambda, T)
        )
    # /* We now have the zodi level in photons/m^2/sr/sec. Convert to photons/m^2/arcsec^2/sec. */
    return zodi_tot / 4.2545170296152206e10


def getBandpasses(
    AB_zeropoint=True,
    default_thin_trunc=True,
    include_all_bands=False,
    sca=None,
    **kwargs,
):
    """Utility to get a dictionary containing the Roman ST bandpasses used for imaging.

    This routine reads in a file containing a list of wavelengths and throughput for all Roman
    bandpasses, and uses the information in the file to create a dictionary. This file is in units
    of effective area (m^2), which includes the nominal mirror size and obscuration in each
    bandpass.  We divide these by the nominal roman.collecting_area, so the bandpass objects
    include both filter transmission losses and the obscuration differences relevant for
    each bandpass.  I.e. you should always use roman.collecting_area for the collecting area
    in any flux calculation, and the bandpass will account for the differences from this.

    In principle it should be possible to replace the version of the file with another one, provided
    that the format obeys the following rules:

    - There is a column called 'Wave', containing the wavelengths in microns.
    - The other columns are labeled by the name of the bandpass.

    The bandpasses can be either truncated or thinned before setting the zero points, by passing in
    the keyword arguments that need to get propagated through to the Bandpass.thin() and/or
    Bandpass.truncate() routines.  Or, if the user wishes to thin and truncate using the defaults
    for those two routines, they can use ``default_thin_trunc=True``.  This option is the default,
    because the stored 'official' versions of the bandpasses cover a wide wavelength range.  So even
    if thinning is not desired, truncation is recommended.

    By default, the routine will set an AB zeropoint (unless ``AB_zeropoint=False``).  The
    zeropoint in GalSim is defined such that the flux is 1 photon/cm^2/sec through the
    bandpass. This differs from an instrumental bandpass, which is typically defined such that the
    flux is 1 photon/sec for that instrument.  The difference between the two can be calculated as
    follows::

        # Shift zeropoint based on effective collecting area in cm^2.
        delta_zp = 2.5 * np.log10(galsim.roman.collecting_area)

    ``delta_zp`` will be a positive number that should be added to the GalSim zeropoints to compare
    with externally calculated instrumental zeropoints.  When using the GalSim zeropoints for
    normalization of fluxes, the ``area`` kwarg to drawImage can be used to get the right
    normalization (giving it the quantity ``galsim.roman.collecting_area``).

    This routine also loads information about sky backgrounds in each filter, to be used by the
    galsim.roman.getSkyLevel() routine.  The sky background information is saved as an attribute in
    each Bandpass object.

    There are some subtle points related to the filter edges, which seem to depend on the field
    angle at some level.  This is more important for the grism than for the imaging, so currently
    this effect is not included in the Roman bandpasses in GalSim.

    The bandpass throughput file is translated from a spreadsheet Roman_effarea_20201130.xlsx at
    https://roman.gsfc.nasa.gov/science/WFI_technical.html.

    Example::

        >>> roman_bandpasses = galsim.roman.getBandpasses()
        >>> f184_bp = roman_bandpasses['F184']

    Parameters:
        AB_zeropoint:       Should the routine set an AB zeropoint before returning the bandpass?
                            If False, then it is up to the user to set a zero point.  [default:
                            True]
        default_thin_trunc: Use the default thinning and truncation options?  Users who wish to
                            use no thinning and truncation of bandpasses, or who want control over
                            the level of thinning and truncation, should have this be False.
                            [default: True]
        include_all_bands:  Should the routine include the non-imaging bands (e.g., grisms)?
                            This does not implement any dispersion physics by itself.
                            There is currently no estimate for the thermal background for these
                            bands and they are set to zero arbitrarily.
                            [default: False]
        sca:             Return the bandpasses dictionary for the particular SCA if given.
                            [default: None]
        **kwargs:           Other kwargs are passed to either `Bandpass.thin` or
                            `Bandpass.truncate` as appropriate.

    @returns A dictionary containing bandpasses for all Roman imaging filters.
    """

    # if sca is None:
    #     # Begin by reading in the file containing the info.
    #     datafile = os.path.join(data_root, "Roman_effarea_20210614.txt")
    #     # One line with the column headings, and the rest as a NumPy array.
    #     data = np.genfromtxt(datafile, names=True)
    # else:
    #     sca_id = "SCA%02d" % (int(sca))

    #     # zfile = zipfile.ZipFile(effarea_zip_file, 'r')
    #     # datafile = zfile.open("Roman_effarea_v8_%s_20240301.ecsv" % (sca_id))
    #     datafile = os.path.join(
    #         effarea_root, "Roman_effarea_v8_%s_20240301.ecsv" % (sca_id)
    #     )
    #     data = ascii.read(datafile)
    #     for index, bp_name in enumerate(data.dtype.names[1:]):
    #         if bp_name in roman2galsim_bandpass:
    #             data.rename_column(bp_name, roman2galsim_bandpass[bp_name])

    data = read_gsfc_effarea(sca=sca)

    wave = 1000.0 * data["Wave"]
    # Read in and manipulate the sky background info.
    sky_file = os.path.join(data_root, "roman_sky_backgrounds.txt")
    sky_data = np.loadtxt(sky_file).transpose()
    ecliptic_lat = sky_data[0, :]
    ecliptic_lon = sky_data[1, :]

    # Parse kwargs for truncation, thinning, etc., and check for nonsense.
    truncate_kwargs = ["blue_limit", "red_limit", "relative_throughput"]
    thin_kwargs = ["rel_err", "trim_zeros", "preserve_range", "fast_search"]
    tmp_truncate_dict = {}
    tmp_thin_dict = {}
    if default_thin_trunc:
        if len(kwargs) > 0:
            galsim_warn(
                "default_thin_trunc is true, but other arguments have been passed"
                " to getBandpasses().  Using the other arguments and ignoring"
                " default_thin_trunc."
            )
            default_thin_trunc = False
    if len(kwargs) > 0:
        for key in list(kwargs.keys()):
            if key in truncate_kwargs:
                tmp_truncate_dict[key] = kwargs.pop(key)
            if key in thin_kwargs:
                tmp_thin_dict[key] = kwargs.pop(key)
        if len(kwargs) != 0:
            raise TypeError("Unknown kwargs: %s" % (" ".join(kwargs.keys())))

    # Set up a dictionary.
    bandpass_dict = {}
    # Loop over the bands.
    for index, bp_name in enumerate(data.dtype.names[1:]):
        if include_all_bands is False and bp_name in non_imaging_bands:
            continue

        # Initialize the bandpass object.
        # Convert effective area units from m^2 to cm^2.
        # Also divide by the nominal Roman collecting area to get a dimensionless throughput.
        bp = Bandpass(
            LookupTable(wave, data[bp_name] * 1.0e4 / collecting_area),
            wave_type="nm",
        )

        # Use any arguments related to truncation, thinning, etc.
        if len(tmp_truncate_dict) > 0 or default_thin_trunc:
            bp = bp.truncate(**tmp_truncate_dict)
        if len(tmp_thin_dict) > 0 or default_thin_trunc:
            bp = bp.thin(**tmp_thin_dict)

        # Set the zeropoint if requested by the user:
        if AB_zeropoint:
            bp = bp.withZeropoint("AB")

        # Store the sky level information as an attribute.
        bp._ecliptic_lat = ecliptic_lat
        bp._ecliptic_lon = ecliptic_lon

        if sca is None:
            bp._sky_level = sky_data[2 + index, :]
        else:
            bp._sky_level = np.zeros_like(sky_data[2 + index, :])
            for i in range(len(ecliptic_lat)):
                bp._sky_level[i] = get_zodi_bkgnd(
                    ecliptic_lat[i],
                    ecliptic_lon[i],
                    0.22,
                    2.5,
                    wave / 1000.0,
                    data[bp_name],
                )

        # Add it to the dictionary.
        bp.name = bp_name if bp_name != "W149" else "W146"
        bandpass_dict[bp.name] = bp

    return bandpass_dict


def read_gsfc_effarea(sca=None, filename=None):
    """Read an effective area file from Roman.

    This just puts together the right invocation to get an Excel-converted
    ECSV file into memory.

    Parameters
    ----------
    sca : int
        the name of the detector. A number between 1-18.
    filename : str
        filename to read in

    Returns
    -------
    astropy.table.Table
        table with effective areas for different Roman bandpasses.
    """
    if filename is None:
        if sca is None:
            filename = os.path.join(data_root, "Roman_effarea_20210614.txt")
        else:
            sca_id = "SCA%02d" % (int(sca))
            filename = os.path.join(
                effarea_root, "Roman_effarea_v8_%s_20240301.ecsv" % (sca_id)
            )

    data = ascii.read(filename)
    for index, bp_name in enumerate(data.dtype.names[1:]):
        if bp_name in roman2galsim_bandpass:
            data.rename_column(bp_name, roman2galsim_bandpass[bp_name])
    return data


def compute_abflux(sca, effarea=None, galsim_filter_name=True):
    """Compute the AB zero point fluxes for each filter.

    How many electrons would a zeroth magnitude AB star deposit in
    Roman's detectors in a second?

    Parameters
    ----------
    sca : int
        the name of the detector. A number between 1-18.
    effarea : astropy.Table.table
        Table from GSFC with effective areas for each filter.

    Returns
    -------
    dict[str] : float
        lookup table of zero point fluxes for each filter (electrons / s)
    """

    if effarea is None:
        effarea = read_gsfc_effarea(sca)

    # get the filter names.  This is a bit ugly since there's also
    # a wavelength column 'Wave', and Excel appends a column to each line
    # which astropy then gives a default name col12 to.
    filter_names = [x for x in effarea.dtype.names if x != "Wave" and "col" not in x]
    abfv = ABZeroSpFluxDens
    out = dict()
    for bandpass in filter_names:
        count_rate = compute_count_rate(
            flux=abfv, bandpass=bandpass, sca=sca, effarea=effarea
        )
        if galsim_filter_name:
            out[bandpass] = count_rate
        else:
            out[galsim2roman_bandpass.get(bandpass, bandpass)] = count_rate


    # Saving the SCA information to use the correct throughput curves for each detector.
    out = {f"SCA{sca:02}": out}
    return out


def compute_count_rate(flux, bandpass, sca, filename=None, effarea=None, wavedist=None):
    """Compute the count rate in a given filter, for a specified SED.

    How many electrons would an object with SED given by
    flux deposit in Roman's detectors in a second?

    Parameters
    ----------
    flux : float or np.ndarray with shape matching wavedist.
        Spectral flux density in units of ergs per second * hertz * cm^2
    bandpass : str
        the name of the bandpass
    sca : int
        the name of the detector. A number between 1-18.
    filename : str
        filename to read in
    effarea : astropy.Table.table
        Table from GSFC with effective areas for each filter.
    wavedist : numpy.ndarray
        Array of wavelengths along which spectral flux densities are defined in microns

    Returns
    -------
    float
        the total bandpass flux (electrons / s)
    """
    # Read in default Roman effective areas from Goddard, if areas not supplied
    if effarea is None:
        effarea = read_gsfc_effarea(sca, filename)

    # If wavelength distribution is supplied, interpolate flux and area
    # over it and the effective area table layout
    if wavedist is not None:
        # Ensure that wavedist and flux have the same shape
        if wavedist.shape != flux.shape:
            raise ValueError("wavedist and flux must have identical shapes!")

        all_wavel = np.unique(np.concatenate((effarea["Wave"], wavedist)))
        all_flux = np.interp(all_wavel, wavedist, flux)
        all_effarea = np.interp(all_wavel, effarea["Wave"], effarea[bandpass])
    else:
        all_wavel = effarea["Wave"]
        all_flux = flux
        all_effarea = effarea[bandpass]

    integrand = all_flux * constants.c / (all_wavel * u.micron) ** 2  # f_lambda
    integrand /= constants.h * constants.c / (all_wavel * u.micron)  # hc/lambda
    integrand *= all_effarea * u.m**2  # effective area in filter
    # integrate.simpson looks like it loses units.  So convert to something
    # we know about.
    integrand = integrand.to(1 / (u.s * u.micron)).value

    zpflux = integrate.simpson(integrand, x=all_wavel)
    # effarea['Wave'] is in microns, so we're left with a number of counts
    # per second

    return zpflux


def get_abflux(bandpass, sca):
    """Get the zero point flux for a particular bandpass.

    This is a simple wrapper for compute_abflux, caching the results.

    Parameters
    ----------
    bandpass : str
        the name of the bandpass
    sca : int
        the name of the detector. A number between 1-18.
    Returns
    -------
    float
        the zero point flux (electrons / s)
    """
    bandpass = galsim2roman_bandpass.get(bandpass, bandpass)

    # If abflux for this bandpass for the specified SCA has been calculated, return the calculated
    # value instead of rerunning an expensive calculation
    abflux = getattr(get_abflux, "abflux", None)
    if (abflux is None) or (f"SCA{sca:02}" not in abflux):
        abflux = compute_abflux(sca, galsim_filter_name=False)
        get_abflux.abflux = abflux
    
    return abflux[f"SCA{sca:02}"][bandpass]
