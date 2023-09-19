"""Parameters class storing a few useful constants for Roman simulations.
"""

import numpy as np
from astropy.time import Time
from astropy import units as u


default_parameters_dictionary = {
    'instrument': {'name': 'WFI',
                   'detector': 'WFI07',
                   'optical_element': 'F184',
                   },
    'exposure': {'start_time': Time('2026-01-01T00:00:00'),
                 'type': 'WFI_IMAGE',
                 'ma_table_number': 1,
                 },
    'pointing': {'ra_v1': 270.0,
                 'dec_v1': 66.0,
                 },
    'wcsinfo': {'ra_ref': 270.0,
                'dec_ref': 66.0,
                'v2_ref': 0,
                'v3_ref': 0,
                'roll_ref': 0,
                },
}


# default read noise
read_noise = 5.0 * u.DN

gain = 1 * u.electron / u.DN

nborder = 4  # number of border pixels used for reference pixels.

ma_table = {1: [[1, 8], [9, 8], [17, 8], [25, 8], [33, 8], [41, 8]],
            2: [[1, 5], [6, 8], [14, 1], [15, 9], [24, 25]],
            3: [[1, 25], [26, 8], [34, 1], [35, 14]]}

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

# V2/V3 coordinates of "center" of WFI array (convention)
v2v3_wficen = (1546.3846181707652, -892.7916365721071)

# default saturation level in DN absent reference file information
saturation = 55000 * u.DN

# arbitrary constant to add to initial L1 image so that pixels aren't clipped at zero.
pedestal = 100 * u.DN

dqbits = dict(saturated=2, jump_det=4, nonlinear=2**16)

NUMBER_OF_DETECTORS = 18

# Radial distance from WFI_CEN to extend object generation in order to cover FOV (degrees)
WFS_FOV = 0.6

# Cosmic Ray defaults
cr = {
    # sample_cr_params
    "min_dEdx": 10,
    "max_dEdx": 10000,
    "min_cr_len": 10,
    "max_cr_len": 2000,
    "grid_size": 10000,
    # simulate_crs
    "flux": 8,
    "area": 16.8,
    "conversion_factor": 0.5,
    "pixel_size": 10,
    "pixel_depth": 5,
}

# Persistence defaults
# delete persistence records fainter than 0.01 electron / s
# e.g., MA table 1 has ~144 s, so this is ~1 electron over the whole exposure.
persistence = {
    # init
    "A": 0.017,
    "x0": 6.0e4,
    "dx": 5.0e4,
    "alpha": 0.045,
    "gamma": 1,
    "half_well": 50000,
    "ignorerate": 0.01
}
