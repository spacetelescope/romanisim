
"""
Unit tests for bandpass functions.  Tested routines:
* read_gsfc_effarea
* compute_abflux
* get_abflux
"""

from romanisim import bandpass
from astropy.table import Table


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


def test_compute_abflux():
    # Test default values
    ab_flux = bandpass.compute_abflux()

    assert ab_flux['F062'] == 49379170641.48872
    assert ab_flux['Grism_1stOrder'] == 75489060948.47092
    assert len(ab_flux.keys()) == 11


def test_get_abflux():
    band = 'F062'

    assert bandpass.get_abflux(band) == 49379170641.48872
