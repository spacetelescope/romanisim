"""
Unit tests for bandpass functions.  Tested routines:
* read_gsfc_effarea
* compute_abflux
* get_abflux
"""

import os
import pytest
from metrics_logger.decorators import metrics_logger
import numpy as np
from romanisim import bandpass
from astropy import constants
from astropy.table import Table
from astropy import units as u
from scipy.stats import norm
from romanisim import log

FILTERLIST = ['F062', 'F158', 'F213']
ABVLIST = [4.938e10, 4.0225e10, 2.55e10]
IFILTLIST = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213'] #, 'F146']

def test_read_gsfc_effarea(tmpdir_factory):
    table_file = str(tmpdir_factory.mktemp("ndata").join("table.csv"))
    data_table = Table()
    data_table['Planet'] = ['Saturn', 'Mars', 'Venus', 'Mercury']
    data_table['Dwarf Planet'] = ['Eris', 'Pluto', 'Makemake', 'Haumeua']
    data_table.write(table_file)

    with open(table_file, 'r') as tmp_file:
        file_data = tmp_file.read()
    with open(table_file, 'w') as tmp_file:
        tmp_file.write("Header Comment line \n" + file_data)

    # Test default table
    read_table = bandpass.read_gsfc_effarea()
    assert read_table['F062'][13] == 0.0052

    # Test imported file
    read_table = bandpass.read_gsfc_effarea(table_file)
    assert read_table['Planet'][1] == 'Mars'
    assert read_table['Dwarf Planet'][2] == 'Makemake'


@pytest.mark.parametrize("filter", FILTERLIST)
def test_compute_abflux(filter):
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
    gauss_flux = bandpass.compute_abflux(data_table)

    # Comparing both fluxes as magnitudes
    assert np.isclose(np.log10(theo_flux.value), np.log10(gauss_flux[filter]), atol=1.0e-6)


@pytest.mark.parametrize("filter, value", zip(FILTERLIST, ABVLIST))
def test_get_abflux(filter, value):
    # Test that proper results (within 10%) are returned for select bands.
    assert np.isclose(bandpass.get_abflux(filter), value, rtol=1.0e-1)


#metrics_logger("DMS233")
#pytest.mark.soctests
def test_convert_flux_to_counts():
    # Define AB zero flux, and dirac delta wavelength
    # abfv = 3631e-23 * u.erg / (u.s * u.cm ** 2 * u.hertz)
    dd_wavel = 1.290 * u.micron
    effarea = bandpass.read_gsfc_effarea()

    # # wavedist = np.linspace(0.4, 2.6, len(area))
    # # thru = norm.pdf(wavedist, wavel.value, 0.01)
    # # thru += 100

    # Create dirac-delta-like distribution for filter response
    # dd_wavedist = np.linspace(dd_wavel.value - 0.001, dd_wavel.value + 0.001, 1000)
    # dd_wavedist = np.linspace(0.4, 2.6, len(bandpass.read_gsfc_effarea()['F129'])) * u.micron
    dd_wavedist = effarea['Wave'] * u.micron
    wave_bin_width = dd_wavedist[1] - dd_wavedist[0]
    # thru = norm.pdf(dd_wavedist, dd_wavel.value, 1)
    thru = norm.pdf(dd_wavedist, dd_wavel.value, 0.001)# * u.erg / (u.s * u.cm ** 2 * u.hertz)

    # Add constant flux
    thru += 100

    # Rescale
    thru *= 1.0e-35

    # Add flux units
    thru *= u.erg / (u.s * u.cm ** 2 * u.hertz)

    theo_flux={}
    theo_flux_sum={}
    gauss_flux={}
    for filter in IFILTLIST:
        # Define filter area
        area = bandpass.read_gsfc_effarea()[filter] * u.m ** 2

        # Analytical flux
        theo_flux[filter] = (wave_bin_width * (np.divide(np.multiply(area, thru), dd_wavedist) / constants.h.to(u.erg * u.s))).to(1 / u.s)

        # Sum the flux in the filter
        theo_flux_sum[filter] = np.sum(theo_flux[filter])
        
        # Computed flux
        gauss_flux[filter] = bandpass.compute_count_rate(thru, filter)

        # Comparing both fluxes
        if filter != 'F129':
            assert np.isclose(theo_flux_sum[filter].value, gauss_flux[filter], atol=1.0e-6)
        else:
            # The granualrity of the binning causes greater difference in the
            # narrow filter containing the dirac delta function
            assert np.isclose(theo_flux_sum[filter].value, gauss_flux[filter], atol=1.0e-4)

    # Create log entry and artifacts
    log.info(f'DMS233: integrated over an input spectra in physical units to derive the number of photons / s.')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'theo_flux': theo_flux,
                   'theo_flux_sum': theo_flux_sum,
                   'gauss_flux': gauss_flux,
                   'thru': thru}
        af.write_to(os.path.join(artifactdir, 'dms233.asdf'))
        

@pytest.mark.parametrize("filter", ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146'])
def test_AB_convert_flux_to_counts(filter):
    # AB Zero Test
    abfv = 3631e-23 * u.erg / (u.s * u.cm ** 2 * u.hertz)

    effarea = bandpass.read_gsfc_effarea()
    wavedist = effarea['Wave'] * u.micron

    thru = abfv * np.ones(len(wavedist))
    gauss_flux = bandpass.compute_count_rate(thru, filter)

    assert np.isclose(bandpass.get_abflux(filter), gauss_flux, atol=1.0e-6)
