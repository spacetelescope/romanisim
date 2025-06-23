"""Parameters class storing useful constants for Roman simulations.
"""

import numpy as np
from astropy.time import Time
from astropy import units as u


# MA tables 1(DEFOCUS_MOD), 2(DEFOCUS_LRG), and 18(DIAGNOSTIC) are excluded

# Rev F
read_pattern = {3: [[1], [2, 3], [4, 5, 6, 7, 8, 9], 
                    [10, 11, 12, 13, 14, 15, 16, 17], [18]],
                4: [[1], [2], [3, 4], [5, 6, 7, 8, 9],
                    [10 + x for x in range(8)],
                    [18 + x for x in range(8)],
                    [26 + x for x in range(8)],
                    [34 + x for x in range(10)], [44]],
                5: [[1], [2], [3, 4], [5, 6, 7, 8, 9, 10],
                    [11 + x for x in range(6)],
                    [17 + x for x in range(6)],
                    [23 + x for x in range(6)],
                    [29 + x for x in range(6)],
                    [35 + x for x in range(8)],
                    [43 + x for x in range(10)],
                    [53 + x for x in range(10)], [63]],
                6: [[1], [2], [3, 4], [5, 6, 7, 8, 9, 10],
                    [11 + x for x in range(6)],
                    [17 + x for x in range(6)],
                    [23 + x for x in range(8)],
                    [31 + x for x in range(8)],
                    [39 + x for x in range(8)],
                    [47 + x for x in range(8)],
                    [55 + x for x in range(8)],
                    [63 + x for x in range(10)],
                    [73 + x for x in range(10)],
                    [83 + x for x in range(10)], [93]],
                7: [[1], [2], [3, 4], 
                    [5 + x for x in range(8)],
                    [13 + x for x in range(8)],
                    [21 + x for x in range(10)],
                    [31 + x for x in range(10)],
                    [41 + x for x in range(12)],
                    [53 + x for x in range(12)],
                    [65 + x for x in range(12)],
                    [77 + x for x in range(12)],
                    [89 + x for x in range(12)],
                    [101 + x for x in range(12)],
                    [113 + x for x in range(12)], [125]],
                8: [[1], [2], [3, 4], 
                    [5 + x for x in range(8)],
                    [13 + x for x in range(9)],
                    [22 + x for x in range(10)],
                    [32 + x for x in range(12)],
                    [44 + x for x in range(16)],
                    [60 + x for x in range(16)],
                    [76 + x for x in range(16)],
                    [92 + x for x in range(16)],
                    [108 + x for x in range(16)], 
                    [124 + x for x in range(16)],
                    [140 + x for x in range(16)], [156]],
                9: [[1], [2], [3, 4], [5 + x for x in range(8)], [13 + x for x in range(8)],
                    [21 + x for x in range(10)],
                    [31 + x for x in range(12)],
                    [43 + x for x in range(16)],
                    [59 + x for x in range(16)],
                    [75 + x for x in range(16)],
                    [91 + x for x in range(16)],
                    [107 + x for x in range(16)],
                    [123 + x for x in range(32)],
                    [155 + x for x in range(32)], [187]],
                10: [[1], [2], [3, 4], [5 + x for x in range(8)],
                    [13 + x for x in range(12)],
                    [25 + x for x in range(16)],
                    [41 + x for x in range(16)],
                    [57 + x for x in range(16)],
                    [73 + x for x in range(16)],
                    [89 + x for x in range(16)],
                    [105 + x for x in range(32)],
                    [137 + x for x in range(32)],
                    [169 + x for x in range(32)], 
                    [201 + x for x in range(32)], [233]],
                11: [[1], [2], [3, 4], 
                     [5 + x for x in range(16)], 
                     [21 + x for x in range(16)], 
                     [37 + x for x in range(16)],
                     [53 + x for x in range(32)], 
                     [85 + x for x in range(32)], 
                     [117 + x for x in range(32)], 
                     [149 + x for x in range(32)], 
                     [181 + x for x in range(32)], 
                     [213 + x for x in range(32)], 
                     [245 + x for x in range(32)], 
                     [277 + x for x in range(32)], [309]],
                12: [[1], [2, 3], [4, 5], [6, 7, 8, 9], 
                     [10, 11, 12, 13], [14, 15, 16, 17], 
                     [18, 19, 20, 21], [22, 23, 24, 25], 
                     [26, 27, 28, 29], [30, 31, 32, 33], 
                     [34, 35, 36, 37], [38, 39, 40, 41], 
                     [42 + x for x in range(8)], 
                     [50 + x for x in range(8)], [58]],
                13: [[1], [2, 3], [4, 5], [6, 7, 8, 9], 
                     [10, 11, 12, 13], [14, 15, 16, 17], 
                     [18, 19, 20, 21], 
                     [22 + x for x in range(8)], 
                     [30 + x for x in range(8)], 
                     [38 + x for x in range(8)], 
                     [46 + x for x in range(8)], 
                     [54 + x for x in range(8)], 
                     [62 + x for x in range(8)], 
                     [70 + x for x in range(8)], [78]],
                14: [[1], [2, 3], [4, 5], [6, 7, 8, 9], 
                     [10, 11, 12, 13, 14, 15], 
                     [16 + x for x in range(8)], 
                     [24 + x for x in range(8)],
                     [32 + x for x in range(8)],
                     [40 + x for x in range(8)],
                     [48 + x for x in range(8)],
                     [56 + x for x in range(8)],
                     [64 + x for x in range(8)],
                     [72 + x for x in range(8)], 
                     [80 + x for x in range(16)],
                     [96]],
                15: [[1], [2, 3], [4, 5],[6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [16, 17, 18, 19, 20, 21],
                     [22 + x for x in range(12)],
                     [34 + x for x in range(12)],
                     [46 + x for x in range(12)],
                     [58 + x for x in range(12)],
                     [70 + x for x in range(12)],
                     [82 + x for x in range(12)],
                     [94 + x for x in range(12)],
                     [106 + x for x in range(12)],
                     [118]],
                16: [[1], [2, 3],[4, 5],
                     [6 + x for x in range(8)],
                     [14 + x for x in range(12)],
                     [26 + x for x in range(12)],
                     [38 + x for x in range(12)],
                     [50 + x for x in range(12)],
                     [62 + x for x in range(12)],
                     [74 + x for x in range(12)],
                     [86 + x for x in range(12)],
                     [98 + x for x in range(16)],
                     [114 + x for x in range(16)],
                     [130 + x for x in range(16)],
                     [146]],
                17: [[1, 2], [3, 4], [5, 6], 
                     [7, 8, 9, 10, 11, 12],
                     [13 + x for x in range(8)],
                     [21 + x for x in range(12)],
                     [33 + x for x in range(16)],
                     [49 + x for x in range(16)],
                     [65 + x for x in range(16)],
                     [81 + x for x in range(16)],
                     [97 + x for x in range(16)],
                     [113 + x for x in range(16)],
                     [129 + x for x in range(32)],
                     [161 + x for x in range(32)],
                     [193]],
                
                # note: these MA tables are not part of the PRD and are intended
                # only to support DMS testing.  In particular, 110 is a
                # 16-resultant imaging table needed to demonstrate ramp fitting
                # performance.
                # both 109 and 110 have CRDS darks for both spectroscopic and
                # imaging modes
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
    'ephemeris': {'time': Time('2026-01-01').mjd,
                  'spatial_x': 0.,
                  'spatial_y': 0.,
                  'spatial_z': 0.,
                  'velocity_x': 0.,
                  'velocity_y': 0.,
                  'velocity_z': 0.
                  },
    'exposure': {'start_time': Time('2026-01-01T00:00:00'),
                 'type': 'WFI_IMAGE',
                 'ma_table_number': 4,
                 'read_pattern': read_pattern[4],
                 # Changing the default MA table to be 4 (C2A_IMG_HLWAS) as MA table 1 (DEFOCUS_MOD) is not supported 
                 },
    'pointing': {'target_ra': 270.0,
                 'target_dec': 66.0,
                 'target_aperture': 'WFI_CEN',
                 'pa_aperture': 0.,
                 },
    'velocity_aberration': {'scale_factor': 1.0},
    'wcsinfo': {'aperture_name': 'WFI_CEN',
                'ra_ref': 270.0,
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

# Default metadata for level 3 mosaics
default_mosaic_parameters_dictionary = {
    'basic': {'time_mean_mjd': Time('2026-01-01T00:00:00').mjd,
              'optical_element': 'F184',
              },
    'wcsinfo': {'ra_ref': 270.0,
                'dec_ref': 66.0,
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
}

nborder = 4  # number of border pixels used for reference pixels.

read_time = 3.16247

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

# Addd this much extra noise as correlated extra noise in all resultants.
pedestal_extra_noise = 4 * u.DN

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

# Centermost PSF to use for mosaic creation
default_sca = 2

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

# angle of V3 relative to +Y
V3IdlYAngle = -60

# fiducial WFI pixel scale in arcseconds
pixel_scale = 0.11
