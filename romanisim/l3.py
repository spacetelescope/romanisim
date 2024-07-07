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


def add_objects_to_l3(l3_mos, source_cat, xpos=None, ypos=None, coords=None,
                      coords_unit='rad', wcs=None, psf=None, rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_mos : MosaicModel
        Mosaic of images
    source_cat : list
        List of catalog objects to add to l3_mos

    Returns
    -------
    None
        l3_mos is updated in place
    """

    # Obtain optical element
    filter_name = l3_mos.meta.basic.optical_element

    if wcs is None:
        # Generate WCS
        wcs = romanisim.wcs.get_mosaic_wcs(l3_mos.meta, shape=l3_mos.data.shape)

    if psf is None:
        # Create PSF
        psf = romanisim.psf.make_psf(filter_name=filter_name, sca=CENTER_SCA, chromatic=False, webbpsf=True)

    if any(pos is None for pos in [xpos,ypos]):
        # Need to construct position arrays
        if coords is None:
            coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                           for o in source_cat])
            coords_unit='rad'
        # Generate x,y positions for sources
        sourcecountsall = galsim.ImageF(l3_mos.data.shape[0], l3_mos.data.shape[1], wcs=wcs, xmin=0, ymin=0)
        xpos, ypos = sourcecountsall.wcs.radecToxy(coords[:, 0], coords[:, 1], coords_unit)

    # if coords is not None:
    #     # Generate x,y positions for sources
    #     sourcecountsall = galsim.ImageF(l3_mos.data.shape[0], l3_mos.data.shape[1], wcs=wcs, xmin=0, ymin=0)
    #     xpos, ypos = sourcecountsall.wcs.radecToxy(coords[:, 0], coords[:, 1], coords_unit)
    # elif all(pos is not None for pos in [xpos,ypos]):
    #     log.error('No (xpos and ypos) or coords set. Adding zero objects.')
    #     return

    print(f"XXX wcs = {wcs}")

    xpos_idx = [round(x) for x in xpos]
    ypos_idx = [round(y) for y in ypos]

    if ((min(xpos_idx) < 0) or (min(ypos_idx) < 0) or (max(xpos_idx) > l3_mos.data.shape[0])
        or (max(ypos_idx) > l3_mos.data.shape[1])):
        log.error(f"A source is out of bounds! "
                  f"Source min x,y = {min(xpos_idx)},{min(ypos_idx)}, max = {max(xpos_idx)},{max(ypos_idx)} "
                  f"Image min x,y = {0},{0}, max = {l3_mos.data.shape[0]},{l3_mos.data.shape[1]}")

    # Create overall scaling factor map
    # Ct_all = (l3_mos.data.value / l3_mos.var_poisson)
    Ct_all = np.divide(l3_mos.data.value, l3_mos.var_poisson.value,
                       out=np.ones(l3_mos.data.shape), where=l3_mos.var_poisson.value != 0)

    Ct = []
    # Cycle over sources and add them to the mosaic
    for idx, (x, y) in enumerate(zip(xpos_idx, ypos_idx)):
        # Set scaling factor for injected sources
        # Flux / sigma_p^2
        if l3_mos.var_poisson[x][y].value != 0:
            Ct.append(math.fabs(l3_mos.data[x][y].value / l3_mos.var_poisson[x][y].value))
        # elif l3_mos.data[x][y].value != 0:
        #     Ct = math.fabs(l3_mos.data[x][y].value)
        else:
            Ct.append(1.0)
  
    # Create empty postage stamp galsim source image
    # sourcecounts = galsim.ImageF(l3_mos.data.shape[0], l3_mos.data.shape[1], wcs=wcs, xmin=0, ymin=0)
    # sourcecounts = galsim.ImageF(l3_mos.data.value, wcs=wcs, xmin=0, ymin=0)
    # sourcecounts = galsim.ImageF(l3_mos.data, wcs=wcs, xmin=0, ymin=0)

    # unit = l3_mos.data.unit

    # Simulate source postage stamp
    # romanisim.image.add_objects_to_image_old(sourcecounts, source_cat, xpos=xpos,
    romanisim.image.add_objects_to_image(l3_mos.data, source_cat, xpos=xpos,
                                            ypos=ypos, psf=psf, flux_to_counts_factor=Ct, exptimes=Ct,
                                            bandpass=[filter_name], filter_name=filter_name, 
                                            wcs=wcs, rng=rng, seed=seed)

    # # Scale the source image back by its flux ratios
    # sourcecounts /= Ct

    # Add sources to the original mosaic data array
    # l3_mos.data = (l3_mos.data.value + sourcecounts) * l3_mos.data.unit
    # l3_mos.data = (l3_mos.data.value + np.swapaxes(sourcecounts.array, 0, 1)) * l3_mos.data.unit
    # print(f"XXX type(l3_mos.data.value) = {type(l3_mos.data.value)}")
    # l3_mos.data = sourcecounts.array * unit
    

    # Note for the future - other noise sources (read and flat) need to be implemented

    # Set new poisson variance
    l3_mos.var_poisson = (l3_mos.data.value / Ct_all) * l3_mos.var_poisson.unit

    # l3_mos is updated in place, so no return
    return None
