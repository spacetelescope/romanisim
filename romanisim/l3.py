"""Roman WFI simulator functions for Level 3 mosaics.

Based on galsim's implementation of Roman image simulation.  Uses galsim Roman modules
for most of the real work.
"""

import numpy as np
import math
import astropy.time
from astropy import table
import galsim
from galsim import roman

from . import parameters
from . import util
import romanisim.wcs
import romanisim.l1
import romanisim.bandpass
import romanisim.psf
import romanisim.image
import romanisim.persistence
from romanisim import log
import roman_datamodels.maker_utils as maker_utils

# Define centermost SCA for PSFs
CENTER_SCA = 2


def add_objects_to_l3(l3_mos, source_cat, exptimes, xpos=None, ypos=None, coords=None, unit_factor=1.0,
                      coords_unit='rad', wcs=None, psf=None, rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_mos : MosaicModel
        Mosaic of images
    source_cat : list
        List of catalog objects to add to l3_mos
    exptimes : list
        Exposure times to scale back to rate units
    xpos, ypos : array_like
        x & y positions of sources (pixel) at which sources should be added
    coords : array_like
        ra & dec positions of sources (coords_unit) at which sources should be added
    unit_factor: float
        Factor to convert data to MJy / sr
    coords_unit : string
        units of coords
    wcs : galsim.GSFitsWCS
        WCS corresponding to image
    psf : galsim.Profile
        PSF for image
    rng : galsim.BaseDeviate
        random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    None
        l3_mos is updated in place
    """

    # Obtain optical element
    filter_name = l3_mos.meta.basic.optical_element

    # Generate WCS (if needed)
    if wcs is None:
        wcs = romanisim.wcs.get_mosaic_wcs(l3_mos.meta, shape=l3_mos.data.shape)

    # Create PSF (if needed)
    if psf is None:
        psf = romanisim.psf.make_psf(filter_name=filter_name, sca=CENTER_SCA, chromatic=False, webbpsf=True)

    # Create Image canvas to add objects to
    sourcecountsall = galsim.ImageF(l3_mos.data.value, wcs=wcs, xmin=0, ymin=0)

    # Create position arrays (if needed)
    if any(pos is None for pos in [xpos, ypos]):
        # Create coordinates (if needed)
        if coords is None:
            coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                               for o in source_cat])
            coords_unit = 'rad'
        # Generate x,y positions for sources
        xpos, ypos = sourcecountsall.wcs.radecToxy(coords[:, 0], coords[:, 1], coords_unit)

    # Add sources to the original mosaic data array
    romanisim.image.add_objects_to_image(sourcecountsall, source_cat, xpos=xpos, ypos=ypos,
                                         psf=psf, flux_to_counts_factor=[xpt * unit_factor for xpt in exptimes],
                                         exptimes=exptimes, bandpass=[filter_name], filter_name=filter_name,
                                         wcs=wcs, rng=rng, seed=seed)

    # Save array with added sources
    l3_mos.data = sourcecountsall.array * l3_mos.data.unit

    # l3_mos is updated in place, so no return
    return None
