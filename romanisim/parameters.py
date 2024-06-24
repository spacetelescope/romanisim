"""Parameters class storing a few useful constants for Roman simulations.
"""

import numpy as np
from astropy.time import Time
from astropy import units as u

read_pattern = {1: [[1 + x for x in range(8)],
                    [9 + x for x in range(8)],
                    [17 + x for x in range(8)],
                    [25 + x for x in range(8)],
                    [33 + x for x in range(8)],
                    [41 + x for x in range(8)]],
                2: [[1 + x for x in range(5)],
                    [6 + x for x in range(8)],
                    [14],
                    [15 + x for x in range(9)],
                    [24 + x for x in range(25)]],
                3: [[1 + x for x in range(25)],
                    [26 + x for x in range(8)],
                    [34],
                    [35 + x for x in range(14)]],
                109: [[1], [2, 3], [5, 6, 7], [10, 11, 12, 13],
                      [15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26],
                      [27, 28, 29, 30, 31, 32], [33, 34, 35, 36, 37, 38],
                      [39, 40, 41, 42, 43], [44]],
                110: [[1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13],
                      [14, 15, 16], [17, 18, 19], [20, 21, 22], [23, 24, 25],
                      [26, 27, 28], [29, 30, 31], [32, 33, 34], [35, 36, 37],
                      [38, 39, 40], [41, 42, 43], [44]],
                }

default_parameters_dictionary = {
    'instrument': {'name': 'WFI',
                   'detector': 'WFI07',
                   'optical_element': 'F184',
                   },
    'exposure': {'start_time': Time('2026-01-01T00:00:00'),
                 'type': 'WFI_IMAGE',
                 'ma_table_number': 1,
                 'read_pattern': read_pattern[1],
                 },
    'pointing': {'ra_v1': 270.0,
                 'dec_v1': 66.0,
                 },
    'wcsinfo': {'ra_ref': 270.0,
                'dec_ref': 66.0,
                'v2_ref': 0,
                'v3_ref': 0,
                'roll_ref': 0,
                'vparity': -1,
                'v3yangle': -60.0,
                # I don't know what vparity and v3yangle should really be,
                # but they are always -1 and -60 in existing files.
                },
    'aperture': {'name': 'WFI_CEN',
                 'position_angle': 0
                 },
}

# Default metadata for level 3 mosaics
default_mosaic_parameters_dictionary = {
    'basic': {'time_mean_mjd': Time('2026-01-01T00:00:00').mjd,
              'optical_element': 'F184',
              },
    'wcsinfo': {'ra_ref': 270.0,
                'dec_ref': 66.0,
                'v2_ref': 0,
                'v3_ref': 0,
                'roll_ref': 0,
                'vparity': -1,
                'v3yangle': -60.0,
                # I don't know what vparity and v3yangle should really be,
                # but they are always -1 and -60 in existing files.
                },
}

reference_data = {
    "dark": 0.01 * u.electron / u.s,
    "distortion": None,
    "flat": None,
    "gain": 2 * u.electron / u.DN,
    "inverselinearity": None,
    "linearity": None,
    "readnoise": 5.0 * u.DN,
    "saturation": 55000 * u.DN,
    "photom": {"F062": 0.28798833 * u.MJy / u.sr,
               "F087": 0.3911463 * u.MJy / u.sr,
               "F106": 0.37039083 * u.MJy / u.sr,
               "F129": 0.36495176 * u.MJy / u.sr,
               "F146": 0.11750617 * u.MJy / u.sr,
               "F158": 0.35352992 * u.MJy / u.sr,
               "F184": 0.53612772 * u.MJy / u.sr,
               "F213": 0.55763922 * u.MJy / u.sr,
               }
}

nborder = 4  # number of border pixels used for reference pixels.

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
v2v3_wficen = (1546.3846181707652, -892.7916365721071)  # arcsec

# persistence parameter dictionary
# delete persistence records fainter than 0.01 electron / s
# e.g., MA table 1 has ~144 s, so this is ~1 electron over the whole exposure.
persistence = dict(A=0.017, x0=6.0e4, dx=5.0e4, alpha=0.045, gamma=1,
                   half_well=50000, ignorerate=0.01)

# arbitrary constant to add to initial L1 image so that pixels aren't clipped at zero.
pedestal = 100 * u.DN

dqbits = dict(saturated=2, jump_det=4, nonlinear=2**16, no_lin_corr=2**20)
dq_do_not_use = dqbits['saturated'] | dqbits['jump_det']

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
