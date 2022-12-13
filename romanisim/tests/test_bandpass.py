"""
Unit tests for bandpass functions.  Tested routines:
* read_gsfc_effarea
* compute_abflux
* get_abflux
"""

import pytest
import numpy as np
from romanisim import bandpass
from astropy import constants
from astropy.table import Table
from astropy import units as u
from scipy.stats import norm

FILTERLIST = ['F062', 'F158', 'F213']
ABVLIST = [4.938e10, 4.0225e10, 2.55e10]


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
