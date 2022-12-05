"""
Unit tests for PSF functions.
"""

import numpy as np
from romanisim import psf
import galsim


class FakeWCS():
    def __init__(self):
        pass

    def toWorld(self, pos):
        return galsim.CelestialCoord(pos.x * 0.1 * galsim.arcsec,
                                     pos.y * 0.1 * galsim.arcsec)


def test_make_psf():
    psfs = []
    psfs.append(psf.make_psf(1, 'F087'))
    psfs.append(psf.make_psf(2, 'F184'))
    psfs.append(psf.make_psf(3, 'F184', webbpsf=False))
    psfs.append(psf.make_psf(4, 'F184', pix=(1000, 1000), webbpsf=False))
    psfs.append(psf.make_psf(5, 'F184', pix=(1000, 1000), webbpsf=True))
    psfs.append(psf.make_psf(6, 'F129', wcs=FakeWCS()))
    for p in psfs:
        im = p.drawImage().array
        totsum = np.sum(im)
        assert totsum < 1
        assert totsum > 0.9
        # assert that image catches no more than 100% and no less than 90%
        # of flux?
        assert np.min(im) > np.max(im) * (-1e-6)
        # ideally nothing negative, though we could loosen this to, say,
        # > -1e-4 * peak and I wouldn't feel too bad.
