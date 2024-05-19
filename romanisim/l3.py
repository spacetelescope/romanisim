"""Function for Level 3-like images.

"""

import numpy as np
import galsim
from . import log
from . import image, wcs, psf


def add_objects_to_l3(l3_mos, source_cat, rng=None, seed=None):
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

    if rng is None and seed is None:
        seed = 143
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    # Obtain optical element
    filter_name = l3_mos.meta.basic.optical_element

    # Generate WCS
    twcs = wcs.get_mosaic_wcs(l3_mos.meta)

    # Create PSF
    l3_psf = psf.make_psf(filter_name=filter_name, sca=2, chromatic=False, webbpsf=True)

    # Generate x,y positions for sources
    coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                       for o in source_cat])
    sourcecountsall = galsim.ImageF(l3_mos.data.shape[0], l3_mos.data.shape[1], wcs=twcs, xmin=0, ymin=0)
    xpos, ypos = sourcecountsall.wcs._xy(coords[:, 0], coords[:, 1])
    xpos = [round(x) for x in xpos]
    ypos = [round(y) for y in ypos]

    # Cycle over sources and add them to the mosaic
    for idx, (x, y) in enumerate(zip(xpos, ypos)):
        # Set scaling factor for injected sources
        # Flux / sigma_p^2
        Ct = (l3_mos.data[x][y].value / l3_mos.var_poisson[x][y].value)

        # Create empty postage stamp galsim source image
        sourcecounts = galsim.ImageF(l3_mos.data.shape[0], l3_mos.data.shape[1], wcs=twcs, xmin=0, ymin=0)

        # Simulate source postage stamp
        image.add_objects_to_image(sourcecounts, [source_cat[idx]], xpos=[x], ypos=[y],
                                   psf=l3_psf, flux_to_counts_factor=Ct, bandpass=[filter_name],
                                   filter_name=filter_name, rng=rng)

        # Scale the source image back by its flux ratios
        sourcecounts /= Ct

        # Add sources to the original mosaic data array
        l3_mos.data = (l3_mos.data.value + sourcecounts.array) * l3_mos.data.unit

    # l3_mos is updated in place, so no return
    return None
