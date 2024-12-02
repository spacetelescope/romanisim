"""
Unit tests for bandpass functions.  Tested routines:
* read_gsfc_effarea
* compute_abflux
* get_abflux
"""

import os
import pytest
import asdf
import numpy as np
from romanisim import bandpass
from astropy import constants
from astropy.table import Table
from astropy import units as u
from scipy.stats import norm, argus
from romanisim import log

# List of all bandpasses for full spectrum tests
IFILTLIST = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']

# List of select filters with calculated AB fluxes for AB Flux test
FILTERLIST = ['F062', 'F158', 'F213']
ABVLIST = [4.938e10, 4.0225e10, 2.55e10]

# Testing with SCA = 1
def test_read_gsfc_effarea(tmpdir_factory, sca=1):
    table_file = str(tmpdir_factory.mktemp("ndata").join("table.ecsv")) # Writing the table_file as ecsv to match the extension of the throughput file
    data_table = Table()
    data_table['Planet'] = ['Saturn', 'Mars', 'Venus', 'Mercury']
    data_table['Dwarf Planet'] = ['Eris', 'Pluto', 'Makemake', 'Haumeua']
    data_table.write(table_file)

    with open(table_file, 'r') as tmp_file:
        file_data = tmp_file.read()
    # Removing the followings as the ECSV files should start with the ECSV version    
    # with open(table_file, 'w') as tmp_file:
    #     tmp_file.write("Header Comment line \n" + file_data)

    # Test default table
    read_table = bandpass.read_gsfc_effarea(sca)
    assert read_table['F062'][13] == 0.0 # Updating the value to match the new table. 0.0052 is the old table value


    # Test imported file
    read_table = bandpass.read_gsfc_effarea(sca, table_file)
    assert read_table['Planet'][1] == 'Mars'
    assert read_table['Dwarf Planet'][2] == 'Makemake'


@pytest.mark.parametrize("filter", IFILTLIST)
def test_compute_abflux(filter, sca=1):
    # Test calculated abfluxes vs analytical values

    # Define AB zero flux, filter area, and wavelength
    abfv = 3631e-23 * u.erg / (u.s * u.cm ** 2 * u.hertz)
    area = 1.0 * u.m ** 2
    wavel = int(filter[1:]) * 0.001 * u.micron

    # Create dirac-delta-like distribution for filter response
    wavedist = np.linspace(wavel.value - 0.001, wavel.value + 0.001, 1000)
    thru = norm.pdf(wavedist, wavel.value, 0.0001)

    # Analytical flux
    theo_flux = (abfv * area / (constants.h.to(u.erg * u.s) * wavel)).to(1 / (u.s * u.micron))

    # Table for filter data storage
    data_table = Table()
    data_table['Wave'] = wavedist
    data_table[filter] = thru

    # Computed flux
    gauss_flux = bandpass.compute_abflux(sca, data_table)

    # Comparing both fluxes as magnitudes
    assert np.isclose(np.log10(theo_flux.value), np.log10(gauss_flux[f'SCA{sca:02}'][filter]), rtol=1.0e-6)


@pytest.mark.parametrize("filter, value", zip(FILTERLIST, ABVLIST))
def test_get_abflux(filter, value, sca=1):
    # Test that proper results (within 10%) are returned for select bands.
    assert np.isclose(bandpass.get_abflux(filter, sca), value, rtol=1.0e-1)


@pytest.mark.soctests
def test_convert_flux_to_counts(sca=1):
    # Define dirac delta wavelength
    dd_wavel = 1.290 * u.micron

    # Define effective area table
    effarea = bandpass.read_gsfc_effarea(sca)

    # Define wavelength distribution
    dd_wavedist = effarea['Wave'] * u.micron

    # Check that the wavelength spacing is constant
    assert np.all(np.isclose(np.diff(dd_wavedist), np.diff(dd_wavedist)[0], rtol=1.0e-6))
    wave_bin_width = np.diff(dd_wavedist)[0]

    # Create dirac-delta-like distribution
    flux = norm.pdf(dd_wavedist, dd_wavel.value, 0.001)

    # Add constant flux
    flux += 100

    # Rescale
    flux *= 1.0e-35

    # Add spectral flux density units
    flux *= u.erg / (u.s * u.cm ** 2 * u.hertz)

    theoretical_flux = {}
    computed_flux = {}
    flux_distro = {}

    for filter in IFILTLIST:
        # Define filter area
        area = bandpass.read_gsfc_effarea(sca)[filter] * u.m ** 2

        # Define pedestal flux
        flux_AB_ratio = ((100.0e-35 * u.erg / (u.s * u.cm ** 2 * u.hertz))
                         / (3631e-23 * u.erg / (u.s * u.cm ** 2 * u.hertz)))
        theoretical_flux[filter] = bandpass.get_abflux(filter, sca) * flux_AB_ratio / u.s

        # Add delta function flux
        dd_flux = (1.0e-35 * u.erg / (u.s * u.cm ** 2 * u.hertz * constants.h.to(u.erg * u.s))
                   * np.interp(1.29, bandpass.read_gsfc_effarea(sca)['Wave'], area) * area.unit).to(1 / u.s)

        theoretical_flux[filter] = theoretical_flux[filter] + dd_flux

        # Computed flux
        computed_flux[filter] = bandpass.compute_count_rate(flux, filter, sca) / u.s

        # Test that proper results (within 0.2%) are returned for select bands.
        assert np.isclose(theoretical_flux[filter].value, computed_flux[filter].value, rtol=2.0e-03)

        # Flux distribution for artifacts
        flux_distro[filter] = (wave_bin_width * (np.divide(np.multiply(area, flux),
                               dd_wavedist) / constants.h.to(u.erg * u.s))).to(1 / u.s)

    # Create log entry and artifacts
    log.info('DMS233: integrated over an input spectra in physical units to derive the number of photons / s.')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'flux_distro': flux_distro,
                   'theoretical_flux': theoretical_flux,
                   'computed_flux': computed_flux,
                   'flux': flux}
        af.write_to(os.path.join(artifactdir, 'dms233.asdf'))


@pytest.mark.parametrize("filter", ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146'])
def test_AB_convert_flux_to_counts(filter, sca=1):
    # AB Zero Test
    abfv = 3631e-23 * u.erg / (u.s * u.cm ** 2 * u.hertz)

    effarea = bandpass.read_gsfc_effarea(sca)
    wavedist = effarea['Wave'] * u.micron

    flux = abfv * np.ones(len(wavedist))
    computed_flux = bandpass.compute_count_rate(flux, filter, sca)

    assert np.isclose(bandpass.get_abflux(filter, sca), computed_flux, rtol=1.0e-6)


def test_unevenly_sampled_wavelengths_flux_to_counts(sca=1):
    # Get filter response table for theoretical curve
    effarea = bandpass.read_gsfc_effarea(sca)

    # Define default wavelength distribution
    wavedist = effarea['Wave'] * u.micron
    wave_bin_width = wavedist[1] - wavedist[0]

    # Define spectral features
    # Linear slope
    flux_slope = np.array([1, 0.5, 0])
    # Curve
    flux_arg = argus.pdf(x=wavedist, chi=1, loc=1, scale=1)
    # Pedestal 1
    flux_flat1 = np.array([0.66, 0.66])
    # Dirac Delta function
    wavedist_dd = np.linspace(2.13 - 0.001, 2.13 + 0.001, 1000)
    flux_dd = norm.pdf(wavedist_dd, 2.13, 0.001)
    # Pedestal 2
    flux_flat2 = np.array([0.33, 0.33])

    # Array to store the theoretical spectral flux density
    an_flux = np.zeros(len(effarea['Wave']))

    # Linear slope from 400nm to 1000nm
    arg_start = np.where(wavedist.value == 1)[0][0]
    an_flux[0:arg_start] = np.arange(start=1, stop=0, step=(-1 / (arg_start)))

    # Define spectral flux array and wavelengths (uneven spacing)
    total_flux = flux_slope.copy()
    total_wavedist = np.array([0.4, 0.7, 1])

    # Curve from 1000nm to 2000nm
    arg_stop = np.where(wavedist.value == 2)[0][0]
    arg_wavedist = wavedist.value[arg_start + 1:arg_stop]
    total_flux = np.append(total_flux, flux_arg[np.where(wavedist.value == 1)[0][0] + 1:
                                                np.where(wavedist.value == 2)[0][0]])
    total_wavedist = np.append(total_wavedist, arg_wavedist)
    an_flux[arg_start:arg_stop + 1] = flux_arg[arg_start:arg_stop + 1]

    # Pedestal from 2000nm to 2130nm
    dd_loc = np.where(wavedist.value == 2.13)[0][0] + 1
    total_flux = np.append(total_flux, flux_flat1)
    total_wavedist = np.append(total_wavedist, np.array([2.0, 2.13 - 0.001]))
    an_flux[arg_stop + 1:dd_loc] = flux_flat1[0]

    # Delta function at 2130nm
    total_flux = np.append(total_flux, flux_dd)
    total_wavedist = np.append(total_wavedist, np.array(wavedist_dd))
    an_flux[dd_loc] = 1 / wave_bin_width.value

    # Pedestal from 2130nm to 2600nm
    total_flux = np.append(total_flux, flux_flat2)
    total_wavedist = np.append(total_wavedist, np.array([2.13 + 0.001, 2.6]))
    an_flux[dd_loc + 1:] = flux_flat2[0]

    # Rescale both spectra
    total_flux *= 1.0e-35
    an_flux *= 1.0e-35

    # Add spectral flux density units
    total_flux *= u.erg / (u.s * u.cm ** 2 * u.hertz)
    an_flux *= u.erg / (u.s * u.cm ** 2 * u.hertz)

    for filter in IFILTLIST:
        # Define filter area
        area = effarea[filter] * u.m ** 2

        # Analytical flux
        an_counts = (wave_bin_width * (np.divide(np.multiply(area, an_flux), wavedist)
                                       / constants.h.to(u.erg * u.s))).to(1 / u.s)

        # Sum the flux in the filter
        an_counts_sum = np.sum(an_counts)

        # Computed flux
        computed_flux = bandpass.compute_count_rate(flux=total_flux, bandpass=filter, sca=sca, wavedist=total_wavedist)

        # Test that proper results (within 4%) are returned for select bands.
        assert np.isclose(an_counts_sum.value, computed_flux, rtol=4.0e-2)
