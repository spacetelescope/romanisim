"""
Unit tests for PSF functions.
"""
import pytest

import numpy as np
from scipy import signal
from romanisim import l1, psf
from romanisim.models import ipc, parameters
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


def requires_unconvolved_reference(model):
    """Skip a test when CRDS is serving a pixel-convolved reference;
    some current tests depend on old versions without pixel convolution.
    """
    if psf.epsf_is_pixel_convolved(model):
        pytest.skip('CRDS is serving a pixel-convolved epsf reference; '
                    'this test is about the older convention')


def test_get_gridded_psf_model_uses_noipc():
    """The gridded PSF model must be built from the IPC-free ``psf_noipc``
    array, not the IPC-convolved ``psf`` array.  romanisim.l1.make_l1 applies
    IPC to the resultants, so using ``psf`` here would convolve IPC twice."""
    focus, spectral_type = 0, 1
    model = psf.get_epsf_from_crds(3, 'F087')
    requires_unconvolved_reference(model)
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
    requires_unconvolved_reference(model)
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


# The oversampled stamp is 4 * 46 + 1 samples on a side, so its center sample
# is index 92 and 92 % OVERSAMPLE == 0.  The samples that fall on native pixel
# centers are then exactly OVERSAMPLE_STRIDE[::OVERSAMPLE], which is what
# test_render_matches_strided_reference compares against.
OVERSAMPLE = 4
STAMP_SIZE = 4 * 46 + 1
GRID_XY = [[0, 0], [0, 4092], [4092, 0], [4092, 4092]]


# EFS: I think we can get rid of this now that some of the other hashing changed?
class Bag:
    """An attribute bag that hashes by identity.

    types.SimpleNamespace would do, except that it defines __eq__ and so is
    unhashable, and romanisim.psf.get_gridded_psf_model is cached.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


ENCLOSED_FLUX = 0.98  # a real stamp loses a little flux off its edge


def make_epsf_reference(pixel_convolved, size=STAMP_SIZE, sigma=1.1):
    """Build a minimal epsf reference model, either pixel-convolved or not.

    The chosen PSF is Gaussian and unrealistic; here we're just testing the
    machinery.  Both the pixel-convolved and unconvolved references are
    built from the same IPC-convolved array
    and differ only by a factor of ``oversample ** 2``, which is what
    `romanisim.psf.epsf_is_pixel_convolved` keys off.
    The pixel-unconvolved reference additionally carries the
    IPC-free ``psf_noipc`` array, which romanisim reads.

    This function also convolves in the IPC into the appropriate arrays,
    'psf' but not 'psf_noipc' for the pixel-unconvolved models.

    Parameters
    ----------
    pixel_convolved : bool
        Which convention to follow.
    size : int
        Samples on a side of the oversampled stamp.
    sigma : float
        Width of the Gaussian, in native pixels.

    Returns
    -------
    An object paralleling roman_datamodels.EpsfRefModel closely enough for
    romanisim.psf.get_gridded_psf_model to work.
    """
    coord = np.arange(size) - (size - 1) / 2
    profile = np.exp(-(coord[:, None] ** 2 + coord[None, :] ** 2)
                     / (2 * (sigma * OVERSAMPLE) ** 2))
    profile *= ENCLOSED_FLUX / profile.sum()
    # (focus, spectral_type, grid position, y, x); get_gridded_psf_model
    # defaults to focus 0 and spectral type 1
    stack = np.tile(profile[None, None, None], (1, 2, len(GRID_XY), 1, 1))

    # IPC couples native pixels, so on the oversampled stamp its nonzero
    # elements are OVERSAMPLE samples apart.
    comb = np.zeros((2 * OVERSAMPLE + 1,) * 2)
    comb[::OVERSAMPLE, ::OVERSAMPLE] = ipc.ipc_kernel
    withipc = signal.convolve(stack, comb[None, None, None], mode='same')

    meta = Bag(
        oversample=OVERSAMPLE,
        pixel_x=Bag(data=Bag(
            data=np.array([xy[0] for xy in GRID_XY], dtype=float))),
        pixel_y=Bag(data=Bag(
            data=np.array([xy[1] for xy in GRID_XY], dtype=float))),
    )

    if pixel_convolved:
        return Bag(psf=withipc * OVERSAMPLE ** 2, meta=meta)
    return Bag(psf=withipc, psf_noipc=stack, meta=meta)


def test_epsf_is_pixel_convolved():
    """The two reference conventions are told apart by their normalization."""
    assert not psf.epsf_is_pixel_convolved(make_epsf_reference(False))
    assert psf.epsf_is_pixel_convolved(make_epsf_reference(True))


def test_gridded_psf_model_normalization():
    """get_gridded_psf_model puts both conventions in galsim's units.

    galsim.InterpolatedImage takes the array sum as the profile's flux, so
    the stamp must sum to the enclosed flux fraction either way: for a
    pixel-convolved reference that means dividing out the oversample ** 2.
    An older reference is already in those units, so it is only a matter of
    reading the IPC-free array rather than the IPC-convolved one, since
    romanisim.l1.make_l1 applies IPC later.
    """
    ref = make_epsf_reference(True)
    gridded = psf.get_gridded_psf_model(ref)
    assert gridded.meta['pixel_convolved']
    np.testing.assert_allclose(np.sum(gridded.data[0]), ENCLOSED_FLUX,
                               rtol=1e-6)

    ref = make_epsf_reference(False)
    gridded = psf.get_gridded_psf_model(ref)
    assert not gridded.meta['pixel_convolved']
    np.testing.assert_array_equal(gridded.data, ref.psf_noipc[0, 1])
    # make sure the IPC convolution did something!
    assert not np.array_equal(ref.psf[0, 1], ref.psf_noipc[0, 1])


def stamp_wcs(wcs, pix, pixel_convolved):
    """The stamp WCS make_one_psf_epsf would build for either convention."""
    if pixel_convolved:
        return psf.psf_stamp_wcs(wcs=wcs, pix=pix, oversample=OVERSAMPLE)
    return psf.psf_stamp_wcs(wcs=wcs, pix=pix,
                             samplescale=parameters.pixel_scale / OVERSAMPLE)


def render_star(ref, wcs, pix=(2044.0, 2044.0), nout=21):
    """Render a unit-flux star centered on a native pixel, as image.py would."""
    gridded = psf.get_gridded_psf_model(ref)
    pixel_convolved = gridded.meta['pixel_convolved']
    stamp = psf.psf_from_grid(gridded, *pix, size=STAMP_SIZE)
    profile = psf.psfstamp_to_galsimimage(stamp, stamp_wcs(wcs, pix,
                                                           pixel_convolved))
    method = 'no_pixel' if pixel_convolved else 'auto'
    return galsim.Convolve(
        galsim.DeltaFunction(flux=1.0), profile).drawImage(
            nx=nout, ny=nout, wcs=wcs.local(), method=method).array


@pytest.mark.parametrize('jacobian', [
    (0.11, 0, 0, 0.11),           # isotropic
    (0.108, 0.012, -0.009, 0.103),  # sheared, like the real distortion
    (0.075, 0.080, -0.078, 0.072),  # rotated
])
def test_render_matches_strided_reference(jacobian):
    """Rendering a star must reproduce the reference's strided samples.

    The defining feature of the pixel-convolved ePSFs is that a star
    rendered at the center of one of the tabulated subpixels is what a
    PSF actually looks like there.  This tests that property.  It's non-trivial
    for romanisim in that the distortion of the PSF WCS must match that of the
    image and the IPC must be successfully deconvolved out of the stamp
    so that it can later be applied.
    """
    ref = make_epsf_reference(True)
    wcs = galsim.JacobianWCS(*jacobian)
    nout = 21
    rendered = render_star(ref, wcs, nout=nout)

    # romanisim deconvolved the IPC out of the reference so that l1 could
    # apply it after the Poisson noise; put it back the way l1 does.
    rendered = signal.convolve(rendered, ipc.ipc_kernel, mode='same')

    center = STAMP_SIZE // 2
    strided = ref.psf[0, 1, 0][center % OVERSAMPLE::OVERSAMPLE,
                               center % OVERSAMPLE::OVERSAMPLE]
    half, mid = nout // 2, strided.shape[0] // 2
    expected = strided[mid - half:mid + half + 1, mid - half:mid + half + 1]

    np.testing.assert_allclose(rendered.sum(), expected.sum(), atol=1e-3)
    assert np.max(np.abs(rendered - expected)) < 1e-3 * expected.max()


def test_pixel_convolved_reference_skips_pixel_convolution():
    """The pixel convolved PSFs look like the pixel-unconvolved PSFs,
    modulo a pixel convolution.

    The same Gaussian, presented under each convention, is rendered with
    'auto' in the un-convolved case, so galsim applies the pixel, and with
    'no_pixel' in the pixel-convolved case, where the reference is taken to
    carry it already.  The rendered images should therefore differ in second
    moment by the variance of a one-pixel top hat.  An isotropic WCS keeps
    the two stamp frames identical so that this is the only difference.

    # EFS: it's not obvious we're doing the IPC handling right here?
    # maybe signal.convolve(rendered, ipc.ipc_kernel, ...) should be present?
    # small effect.
    """
    wcs = galsim.JacobianWCS(parameters.pixel_scale, 0, 0,
                             parameters.pixel_scale)
    optical = render_star(make_epsf_reference(False), wcs, nout=41)
    convolved = render_star(make_epsf_reference(True), wcs, nout=41)

    def second_moment(image):
        y, x = np.mgrid[:image.shape[0], :image.shape[1]]
        cx = (x * image).sum() / image.sum()
        return ((x - cx) ** 2 * image).sum() / image.sum()

    assert second_moment(optical) > second_moment(convolved)
    np.testing.assert_allclose(
        second_moment(optical) - second_moment(convolved), 1 / 12, rtol=0.02)


@pytest.mark.parametrize('size', [STAMP_SIZE, STAMP_SIZE - 1])
def test_deconvolve_ipc_round_trip(size):
    """Deconvolving IPC here and reapplying it in l1 must cancel exactly.

    romanisim removes the IPC baked into a pixel-convolved reference so that
    romanisim.l1 can reapply it later to correlate the Poisson noise.
    This deconvolution / convolution must round-trip for even and odd sized
    stamps.
    """
    ref = make_epsf_reference(True, size=size)
    assert ref.psf.shape[-1] % 2 == size % 2

    gridded = psf.get_gridded_psf_model(ref)
    # l1 applies the kernel on the native grid; on the oversampled stamp that
    # is the same convolution with the taps oversample samples apart.
    reconvolved = gridded.data[:1].copy()
    kernel = np.zeros((2 * OVERSAMPLE + 1,) * 2)
    kernel[::OVERSAMPLE, ::OVERSAMPLE] = ipc.ipc_kernel
    l1.add_ipc(reconvolved, ipc_kernel=kernel, mode='constant', cval=0)

    expected = ref.psf[0, 1, 0] / OVERSAMPLE ** 2
    interior = slice(2 * OVERSAMPLE, -2 * OVERSAMPLE)
    assert np.max(np.abs(reconvolved[0] - expected)[interior, interior]) < (
        1e-10 * expected.max())


def test_pixel_convolved_psf_carries_distortion():
    """A native-pixel reference is placed on the detector grid, not a square one.

    The stamp axes are detector axes, so the full local Jacobian applies and
    the PSF picks up the plate scale and shear that hold at this spot on the
    focal plane.  An older reference is on an idealized grid of square
    pixel_scale pixels, and keeps only the local rotation.
    """
    wcs = galsim.JacobianWCS(0.108, 0.012, -0.009, 0.103)
    pix = (2044.0, 2044.0)

    matrix = np.array(stamp_wcs(wcs, pix, True).getMatrix())
    np.testing.assert_allclose(matrix, wcs.local().getMatrix() / OVERSAMPLE,
                               rtol=1e-10)

    # the older convention: an isotropic grid, so equal scales and no shear
    scales = np.linalg.svd(
        np.array(stamp_wcs(wcs, pix, False).getMatrix()), compute_uv=False)
    np.testing.assert_allclose(scales[0], scales[1], rtol=1e-10)
    np.testing.assert_allclose(scales[0],
                               parameters.pixel_scale / OVERSAMPLE, rtol=1e-10)
