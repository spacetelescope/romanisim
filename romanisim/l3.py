"""Function for Level 3-like images.

"""

import numpy as np
import galsim
from . import log
from . import image, wcs, psf


def add_objects_to_l3(l3_img, source_cat, rng=None, seed=None):
    """Add objects to a Level 3 mosaic

    Parameters
    ----------
    l3_img : MosaicModel
        Mosaic of images
    source_cat : list
        List of catalog objects to add to l3_img

    Returns
    -------
    None
        l3_img is updated in place
    """

    if rng is None and seed is None:
        seed = 143
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.UniformDeviate(seed)

    # Obtain observation keywords
    filter_name = l3_img.meta.basic.optical_element

    # Generate WCS
    twcs = wcs.get_mosaic_wcs(l3_img.meta)

    # Create PSF
    l3_psf = psf.make_psf(filter_name=filter_name, sca=2, chromatic=False, webbpsf=True)

    # Generate x,y positions for sources
    coords = np.array([[o.sky_pos.ra.rad, o.sky_pos.dec.rad]
                       for o in source_cat])
    sourcecountsall = galsim.ImageF(l3_img.data.shape[0], l3_img.data.shape[1], wcs=twcs, xmin=0, ymin=0)
    xpos, ypos = sourcecountsall.wcs._xy(coords[:, 0], coords[:, 1])
    xpos = [round(x) for x in xpos]
    ypos = [round(y) for y in ypos]

    # Cycle over sources and add them to the image
    for idx, (x, y) in enumerate(zip(xpos, ypos)):
        # Set scaling factor for injected sources
        # Flux / sigma_p^2
        Ct = (l3_img.data[x][y].value / l3_img.var_poisson[x][y].value)

        # Create empty postage stamp galsim source image
        sourcecounts = galsim.ImageF(l3_img.data.shape[0], l3_img.data.shape[1], wcs=twcs, xmin=0, ymin=0)

        # Simulate source postage stamp
        image.add_objects_to_image(sourcecounts, [source_cat[idx]], xpos=[x], ypos=[y],
                                   psf=l3_psf, flux_to_counts_factor=Ct, bandpass=[filter_name],
                                   filter_name=filter_name, rng=rng)

        # Scale the source image back by its flux ratios
        sourcecounts /= Ct

        # Add sources to the original image array
        l3_img.data = (l3_img.data.value + sourcecounts.array) * l3_img.data.unit

    # l3_img is updated in place, so no return
    return None
