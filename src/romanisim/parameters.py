"""Parameters class storing a few useful constants for Roman simulations.
"""

default_parameters_dictionary = {
    'roman.meta.instrument.name': 'WFI',
    'roman.meta.instrument.detector': 'WFI07',
    'roman.meta.exposure.start_time':'2026-01-01T00:00:00.000',
    'roman.meta.exposure.type': 'WFI_IMAGE',
    'roman.meta.exposure.ma_table_number': 1,
    'roman.meta.instrument.optical_element': 'F184',
    'roman.meta.pointing.ra_v1': 270.0,
    'roman.meta.pointing.dec_v1': 66.0,
}

read_noise = 5.0
# grabbing the median of the read noise image from CRDS at
# some point

nborder = 4  # number of border pixels used for reference pixels.

ma_table = {1: [[1, 8], [9, 8], [17, 8], [25, 8], [33, 8], [41, 8]]}

read_time = 3.04
