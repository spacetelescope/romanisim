"""
Unit tests for PSF functions.
"""
import pytest

import numpy as np
from romanisim import psf
from romanisim.models.bandpass import galsim2roman_bandpass
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
    ((3, 'F087'), {'psftype': 'epsf'}, None),
    ((4, 'F184'), {'psftype': 'galsim'}, None),
    ((5, 'H158'), {'psftype': 'galsim'}, None),
    ((6, 'H158'), {'psftype': 'galsim', 'chromatic': True}, None),
    ((7, 'F184'), {'pix': (1000, 1000), 'psftype': 'galsim'}, None),
    ((8, 'F184'), {'pix': (1000, 1000), 'psftype': 'stpsf', 'nlambda': 1}, None),
    ((9, 'F184'), {'pix': (1000, 1000), 'psftype': 'epsf'}, None),
    ((10, 'F129'), {'psftype': 'stpsf', 'wcs': FakeWCS(), 'nlambda': 1}, None),
    ((11, 'F087'), {'psftype': 'stpsf', 'variable': True, 'nlambda': 1}, (100, 100)),
    ((12, 'F129'), {'psftype': 'epsf', 'wcs': FakeWCS()}, None),
    ((13, 'F087'), {'psftype': 'epsf', 'variable': True}, (100, 100))])
def test_make_psf(args, kwargs, position):
    p = psf.make_psf(*args, **kwargs)
    if position is not None:
        p = p.at_position(*position)

    bandpass = galsim.roman.getBandpasses(AB_zeropoint=True)['H158']
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')

    if not kwargs.get('chromatic', False):
        method = 'auto'
        im = p.drawImage(method=method).array
    else:
        im = (p * vega_sed.withFlux(1, bandpass)).drawImage(bandpass).array
    totsum = np.sum(im)
    assert totsum < 1
    assert totsum > 0.9
    # assert that image catches no more than 100% and no less than 90%
    # of flux?
    assert np.min(im) > np.max(im) * (-1e-3)
    # ideally nothing negative


@pytest.mark.parametrize("sca", [1, 2])
def test_get_epsf_from_crds_detector_match(sca):
    """get_epsf_from_crds must request the SCA-specific CRDS detector
    (WFIxx); a wrong detector value (e.g. SCAxx) fails to match any
    SCA-specific epsf rmap entry and CRDS silently falls back to the
    same generic, non-SCA-specific reference file for every SCA."""
    filter_name = 'F087'
    model = psf.get_epsf_from_crds(sca, filter_name)
    assert model.meta.instrument.detector == f'WFI{sca:02d}'
    assert model.meta.instrument.optical_element == galsim2roman_bandpass[filter_name]


def test_get_epsf_from_crds_detector_varies_by_sca():
    """Different SCAs must resolve to different epsf reference files;
    if the CRDS detector header were wrong, every SCA would collapse
    onto the same fallback reference."""
    filter_name = 'F087'
    model1 = psf.get_epsf_from_crds(1, filter_name)
    model2 = psf.get_epsf_from_crds(2, filter_name)
    assert model1.meta.instrument.detector != model2.meta.instrument.detector


def test_get_gridded_psf_model_uses_noipc():
    """The gridded PSF model must be built from the IPC-free ``psf_noipc``
    array, not the IPC-convolved ``psf`` array.  romanisim.l1.make_l1 applies
    IPC to the resultants, so using ``psf`` here would convolve IPC twice."""
    focus, spectral_type = 0, 1
    model = psf.get_epsf_from_crds(3, 'F087')
    gridded = psf.get_gridded_psf_model(
        model, focus=focus, spectral_type=spectral_type)

    noipc = np.asarray(model.psf_noipc[focus, spectral_type])
    withipc = np.asarray(model.psf[focus, spectral_type])

    np.testing.assert_array_equal(gridded.data, noipc)
    # guard against the reference file shipping identical arrays, which would
    # make the check above pass vacuously
    assert not np.array_equal(noipc, withipc)


@pytest.mark.xfail(strict=True, reason=(
    'The CRDS ePSF reference is incorrect; the psf extension is built '
    'from the noipc extension by directly convolving with the IPC '
    'kernel rather than respecting the different sampling, and '
    'loses 0.5% of the flux.  See #382.'))
def test_epsf_noipc_plus_ipc_matches_psf():
    """The reference ``psf`` array should be ``psf_noipc`` with IPC applied.

    IPC couples native detector pixels, so on an oversampled stamp it couples
    pixels separated by the oversampling.  This test verifies that the
    IPC-convolved ePSF matches this expectation.
    """
    from scipy import ndimage
    from romanisim.models.ipc import ipc_kernel

    focus, spectral_type, grid_index = 0, 1, 4
    model = psf.get_epsf_from_crds(3, 'F087')
    oversample = model.meta.oversample

    noipc = np.asarray(model.psf_noipc[focus, spectral_type, grid_index],
                       dtype=np.float64)
    withipc = np.asarray(model.psf[focus, spectral_type, grid_index],
                         dtype=np.float64)

    # IPC on native pixels, expressed on the oversampled grid: the 3x3 kernel
    # linking only subpixels oversample apart.
    kernel = np.zeros((2 * oversample + 1, 2 * oversample + 1))
    kernel[::oversample, ::oversample] = ipc_kernel

    convolved = ndimage.convolve(noipc, kernel, mode='constant', cval=0)

    # the kernel sums to one, so IPC redistributes flux without destroying it
    assert np.isclose(withipc.sum(), noipc.sum(), rtol=1e-5)

    resid = np.max(np.abs(convolved - withipc)) / np.max(withipc)
    assert resid < 1e-6
