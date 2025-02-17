"""
Unit tests for PSF functions.
"""

import numpy as np
from romanisim import psf
import galsim
import galsim.roman


class FakeWCS():
    def __init__(self):
        pass

    def toWorld(self, pos):
        return galsim.CelestialCoord(pos.x * 0.1 * galsim.arcsec,
                                     pos.y * 0.1 * galsim.arcsec)

    def local(self, *args, **kwargs):
        return galsim.JacobianWCS(0.1, 0, 0, 0.1)


def test_make_psf():
    psfs = []
    pkw = dict(nlambda=1)
    psfs.append(psf.make_psf(1, 'F087', **pkw))
    psfs.append(psf.make_psf(2, 'F184', chromatic=False, **pkw))
    psfs.append(psf.make_psf(3, 'F184', stpsf=False))
    psfs.append(psf.make_psf(4, 'H158', stpsf=False, chromatic=True))
    psfs.append(psf.make_psf(5, 'F184', pix=(1000, 1000), stpsf=False))
    psfs.append(psf.make_psf(6, 'F184', pix=(1000, 1000), stpsf=True,
                             **pkw))
    psfs.append(psf.make_psf(7, 'F129', wcs=FakeWCS(), **pkw))
    chromatic = [False] * 7
    chromatic[3] = True
    bandpass = galsim.roman.getBandpasses(AB_zeropoint=True)['H158']
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
    varpsf = psf.make_psf(8, 'F087', stpsf=True, variable=True, **pkw)
    psfs.append(varpsf.at_position(100, 100))
    for p, c in zip(psfs, chromatic):
        if not c:
            im = p.drawImage().array
        else:
            im = (p * vega_sed.withFlux(1, bandpass)).drawImage(bandpass).array
        totsum = np.sum(im)
        assert totsum < 1
        assert totsum > 0.9
        # assert that image catches no more than 100% and no less than 90%
        # of flux?
        assert np.min(im) > np.max(im) * (-1e-3)
        # ideally nothing negative
