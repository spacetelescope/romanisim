"""Parameters class storing a few useful constants for Roman simulations.
"""

import numpy as np
from roman_datamodels import units as ru


default_parameters_dictionary = {
    'roman.meta.instrument.name': 'WFI',
    'roman.meta.instrument.detector': 'WFI07',
    'roman.meta.exposure.start_time': '2026-01-01T00:00:00.000',
    'roman.meta.exposure.type': 'WFI_IMAGE',
    'roman.meta.exposure.ma_table_number': 1,
    'roman.meta.instrument.optical_element': 'F184',
    'roman.meta.pointing.ra_v1': 270.0,
    'roman.meta.pointing.dec_v1': 66.0,
    'roman.meta.wcsinfo.ra_ref': 270.0,
    'roman.meta.wcsinfo.dec_ref': 66.0,
    'roman.meta.wcsinfo.v2_ref': 0,
    'roman.meta.wcsinfo.v3_ref': 0,
    'roman.meta.wcsinfo.roll_ref': 0,
}

read_noise = 5.0 * ru.DN
# grabbing the median of the read noise image from CRDS at
# some point

gain = 1 * ru.electron / ru.DN

nborder = 4  # number of border pixels used for reference pixels.

ma_table = {1: [[1, 8], [9, 8], [17, 8], [25, 8], [33, 8], [41, 8]]}

read_time = 3.04

# from draft wfisim documentation
# assumed to be compatible in direction with scipy.ndimage.convolve.
# this is consistent with Andrea Bellini's convention, where, in the
# following kernel, 0.2% of the flux is redistributed to the pixel
# that is +1 - N spaces ahead in memory.
ipc_kernel = np.array(
    [[0.21, 1.66, 0.22],
     [1.88, 91.59, 1.87],
     [0.21, 1.62, 0.2]])
ipc_kernel /= np.sum(ipc_kernel)

v2v3_wficen = (1546.3846181707652, -892.7916365721071)
