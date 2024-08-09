"""Roman WFI simulator functions for Level 3 mosaics.

Based on galsim's implementation of Roman image simulation.  Uses galsim Roman modules
for most of the real work.
"""

import numpy as np
import galsim

from . import parameters
import romanisim.wcs
import romanisim.l1
import romanisim.bandpass
import romanisim.psf
import romanisim.image
import romanisim.persistence
import roman_datamodels.datamodels as rdm
from roman_datamodels.stnode import WfiMosaic


def add_objects_to_l3(l3_mos, source_cat, exptimes, xpos=None, ypos=None, coords=None, cps_conv=1.0, unit_factor=1.0,
                      filter_name=None, coords_unit='rad', wcs=None, psf=None, rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_mos : MosaicModel or galsim.Image
        Mosaic of images
    source_cat : list
        List of catalog objects to add to l3_mos
    exptimes : list
        Exposure times to scale back to rate units
    xpos, ypos : array_like
        x & y positions of sources (pixel) at which sources should be added
    coords : array_like
        ra & dec positions of sources (coords_unit) at which sources should be added
    cps_conv : float
        Factor to convert data to cps
    unit_factor: float
        Factor to convert counts data to MJy / sr
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
    if filter_name is None:
        filter_name = l3_mos.meta.basic.optical_element

    # Generate WCS (if needed)
    if wcs is None:
        wcs = romanisim.wcs.get_mosaic_wcs(l3_mos.meta, shape=l3_mos.data.shape)

    # Create PSF (if needed)
    if psf is None:
        psf = romanisim.psf.make_psf(filter_name=filter_name, sca=parameters.default_sca, chromatic=False, webbpsf=True)

    # Create Image canvas to add objects to
    if isinstance(l3_mos, (rdm.MosaicModel, WfiMosaic)):
        sourcecountsall = galsim.ImageF(l3_mos.data.value, wcs=wcs, xmin=0, ymin=0)
    else:
        sourcecountsall = l3_mos

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
                                         psf=psf, flux_to_counts_factor=[xpt * cps_conv for xpt in exptimes],
                                         convtimes=[xpt / unit_factor for xpt in exptimes],
                                         bandpass=[filter_name], filter_name=filter_name,
                                         rng=rng, seed=seed)

    # Save array with added sources
    if isinstance(l3_mos, (rdm.MosaicModel, WfiMosaic)):
        l3_mos.data = sourcecountsall.array * l3_mos.data.unit

    # l3_mos is updated in place, so no return
    return None
