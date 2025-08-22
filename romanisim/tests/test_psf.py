"""
Unit tests for PSF functions.
"""
import pytest

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


@pytest.mark.parametrize("args, kwargs, position", [
    ((1, 'F087'), {'psftype': 'stpsf', 'nlambda': 1}, None),
    ((2, 'F184'), {'psftype': 'stpsf', 'nlambda': 1}, None),
    ((3, 'F184'), {'psftype': None}, None),
    ((4, 'H158'), {'psftype': None}, None),
    ((5, 'F184'), {'pix': (1000, 1000), 'psftype': None}, None),
    ((6, 'F184'), {'pix': (1000, 1000), 'psftype': 'stpsf', 'nlambda': 1}, None),
    ((7, 'F129'), {'psftype': 'stpsf', 'wcs': FakeWCS(), 'nlambda': 1}, None),
    ((8, 'F087'), {'psftype': 'crds', 'nlambda': 1}, None),
    ((9, 'F087'), {'psftype': 'stpsf', 'variable': True, 'nlambda': 1}, (100, 100)),
])
def test_make_psf(args, kwargs, position):
    p = psf.make_psf(*args, **kwargs)
    if position is not None:
        p = p.at_position(*position)

    bandpass = galsim.roman.getBandpasses(AB_zeropoint=True)['H158']
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')

    if not kwargs.get('chromatic', False):
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
