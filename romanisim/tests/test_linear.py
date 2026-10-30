"""Unit tests for linear module.

Routines tested:
- repair_coefficients
- evaluate_nl_polynomial
- apply
"""
import numpy as np
import galsim
import pytest
from astropy import stats
import crds

import roman_datamodels
from romanisim.models import parameters
from romanisim import log
from romanisim.models import nonlinearity


def test_linear_apply():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.array([0, 0.994, 3.0e-5, 5.0e-10, 7.0e-15], dtype='f4')
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    gain = 4.0  # electron/DN
    counts[0, 0] = counts[0, 99] = counts[99, 0] = counts[99, 99] = 11.0

    linearity = nonlinearity.Nonlinearity(coeffs=lin_coeffs, gain=gain)

    res2 = linearity.apply(counts, electrons=True)

    assert res2[0, 0] == res2[0, 99]
    assert res2[99, 0] == res2[99, 99]
    assert res2[0, 0] == 2 * res2[99, 99]


def test_repair_coeffs():
    counts = np.random.poisson(100, size=(100, 100))

    coeffs = np.array([0, 0.994, 3.0e-5, 5.0e-10, 7.0e-15], dtype='f4')
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0

    # Assign invalid coefficients to be repaired (no correction applied to pixels).
    lin_coeffs[:, 1, 1] *= 0
    lin_coeffs[2, 22, 22] = np.nan

    gain = 4.0  # electron/DN

    linearity = nonlinearity.Nonlinearity(coeffs=lin_coeffs, gain=gain, getdq=True)

    assert linearity.dq[1, 1] == parameters.dqbits['no_lin_corr']
    assert linearity.dq[22, 22] == parameters.dqbits['no_lin_corr']
    # All other entries should be zero
    assert np.count_nonzero(linearity.dq) == 2

    res = linearity.apply(counts)

    assert res[1, 1] == counts[1, 1]
    assert res[22, 22] == counts[22, 22]
    # All other entries should be the same
    assert np.sum(res != counts) == np.prod(counts.shape) - 2


def test_electrons():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.array([0, 0.994, 3.0e-5, 5.0e-10, 7.0e-15], dtype='f4')
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    gain = 4.0  # electron/DN

    linearity = nonlinearity.Nonlinearity(coeffs=lin_coeffs, gain=gain)

    res = linearity.apply(counts)  # DN

    res_elec = linearity.apply(gain * counts, electrons=True)  # electrons

    assert np.all(res_elec[:] == gain * res[:])


def test_reverse():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.array([0, 0.994, 3.0e-5, 5.0e-10, 7.0e-15], dtype='f4')
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))

    lin_coeffs[:, 0:50, :] *= 2.0
    rev_lin_coeffs = lin_coeffs[::-1, ...]
    gain = 4.0  # electron/DN

    linearity = nonlinearity.Nonlinearity(coeffs=lin_coeffs, gain=gain)
    rev_linearity = nonlinearity.Nonlinearity(coeffs=rev_lin_coeffs, gain=gain)

    res = linearity.apply(counts)
    res_rev = rev_linearity.apply(counts, reversed=True)

    assert np.all(res_rev[:] == res[:])


def test_inl_correction():
    """Test integral nonlinearity correction with a simple +1 correction."""
    from types import SimpleNamespace

    # Simulate a trimmed 2-channel image: 2*128 = 256 original columns,
    # minus 2*nborder border pixels = 248 trimmed columns.
    nborder = parameters.nborder
    ncols = 256 - 2 * nborder
    inl_model = SimpleNamespace(
        value=np.arange(65536, dtype='f4'),
        inl_table=SimpleNamespace()
    )
    for channel_num in (1, 2):
        attr_name = f"science_channel_{channel_num:02d}"
        setattr(inl_model.inl_table, attr_name,
                SimpleNamespace(correction=np.ones(65536, dtype='f4')))

    # Identity polynomial coefficients (output = input)
    identity_coeffs = np.zeros((5, 100, ncols), dtype='f4')
    identity_coeffs[1, :, :] = 1.0  # linear term = 1

    # Test forward (linearity) adds +1
    linearity = nonlinearity.Nonlinearity(
        coeffs=identity_coeffs, gain=1.0, integralnonlinearity=inl_model,
        reftype="linearity"
    )
    counts = np.ones((100, ncols), dtype='f4') * 1000
    result = linearity.apply(counts)
    np.testing.assert_allclose(result, counts + 1)

    # Test inverse subtracts 1 (correction is negated)
    inv_linearity = nonlinearity.Nonlinearity(
        coeffs=identity_coeffs, gain=1.0, integralnonlinearity=inl_model,
        reftype='inverselinearity'
    )
    result_inv = inv_linearity.apply(counts)
    np.testing.assert_allclose(result_inv, counts - 1)

    # Test round-trip: apply inverse then forward recovers original
    round_trip = linearity.apply(inv_linearity.apply(counts))
    np.testing.assert_allclose(round_trip, counts)


def test_inverse_then_linearity():
    # Test that applying inverse linearity and then linearity returns the results to
    # the original value

    reffiles = crds.getreferences(
        {'roman.meta.instrument.name': 'WFI',
         'roman.meta.instrument.detector': 'WFI01',
         'roman.meta.exposure.start_time': '2026-01-01T00:00:00'},
        reftypes=['inverselinearity', 'linearity'],
        observatory='roman')

    inverse_linearity_model = roman_datamodels.datamodels.InverselinearityRefModel(
        reffiles['inverselinearity'])
    linearity_model = roman_datamodels.datamodels.LinearityRefModel(
        reffiles['linearity'])

    inverse_linearity = nonlinearity.Nonlinearity(
        coeffs=inverse_linearity_model.coeffs[:, 4:-4, 4:-4], gain=1.0)
    linearity = nonlinearity.Nonlinearity(
        coeffs=linearity_model.coeffs[:, 4:-4, 4:-4], gain=1.0)

    # identify problematic linearity fits
    m = ((linearity_model.dq[4:-4, 4:-4] != 0)
         | (inverse_linearity_model.dq[4:-4, 4:-4] != 0))
    m |= np.all(linearity_model.coeffs[:, 4:-4, 4:-4] == 0, axis=0)

    counts = np.random.poisson(16000, size=(4088, 4088))
    level_0_inv = inverse_linearity.apply(counts)
    level_0_lin = linearity.apply(level_0_inv)

    badfrac = np.sum(m) / np.prod(m.shape)
    if badfrac > 0.1:
        log.info(f'{badfrac:5.1%} pixels have problematic linearity coeffs.')

    rms = stats.mad_std(counts[~m] / level_0_lin[~m] - 1)

    assert rms < 2e-3

    # refined by Newton iteration, the inverse is exact
    exact = nonlinearity.Nonlinearity(
        coeffs=inverse_linearity_model.coeffs[:, 4:-4, 4:-4], gain=1.0,
        inverse_of=linearity)
    err = np.abs(linearity.apply(exact.apply(counts)) - counts)[~m]
    assert np.percentile(err, 99.99) < 1e-3
    # never worse; a few unflagged pixels have polynomials far off
    assert np.all(err <= np.abs(level_0_lin - counts)[~m])


def _jagged_inl(rng, scale=0.02):
    """INL reference with smooth structure plus per-DN jaggedness, with
    slopes of a few percent per DN like the CRDS references."""
    from types import SimpleNamespace
    value = np.arange(65536, dtype='u2')
    inl_model = SimpleNamespace(value=value, inl_table=SimpleNamespace())
    for channel_num in (1, 2):
        corr = (5 * np.sin(2 * np.pi * value / (1000 * channel_num))
                + scale * rng.standard_normal(value.size)).astype('f4')
        setattr(inl_model.inl_table, f"science_channel_{channel_num:02d}",
                SimpleNamespace(correction=corr))
    return inl_model


def test_inverse_of():
    """With inverse_of, the inverse linearity undoes the linearity exactly
    although its polynomial is only an approximate inverse."""
    rng = np.random.default_rng(1)
    ncols = 256 - 2 * parameters.nborder  # two INL channels
    inl = _jagged_inl(rng)
    # linearity y + a y^2 + b y^3 (~5% at 50000 DN); the inverse is its series
    # truncated after x^3, tens of DN off like the CRDS pairs
    a = rng.uniform(0.8e-6, 1.2e-6, (50, ncols))
    b = 1e-12
    lin_coeffs = np.array([0 * a, 1 + 0 * a, a, b + 0 * a], dtype='f4')
    inv_coeffs = np.array([0 * a, 1 + 0 * a, -a, 2 * a ** 2 - b], dtype='f4')
    gain = rng.uniform(1.6, 2.1, lin_coeffs.shape[1:])
    lin_coeffs[:, 0, 0] = [100, 0, 0, 0]  # a pixel that cannot be inverted
    linearity = nonlinearity.Nonlinearity(
        coeffs=lin_coeffs, gain=gain, integralnonlinearity=inl,
        reftype='linearity')
    kw = dict(coeffs=inv_coeffs, gain=gain, integralnonlinearity=inl,
              reftype='inverselinearity')
    approx = nonlinearity.Nonlinearity(**kw)
    exact = nonlinearity.Nonlinearity(**kw, inverse_of=linearity)

    counts = rng.uniform(3000, 50000, lin_coeffs.shape[1:])  # DN
    err_approx = np.abs(linearity.apply(approx.apply(counts)) - counts)
    err_exact = np.abs(linearity.apply(exact.apply(counts)) - counts)
    assert np.median(err_approx) > 1
    assert np.max(err_exact[1:]) < 0.05
    # the uninvertible pixel keeps the polynomial result
    assert exact.apply(counts)[0, 0] == approx.apply(counts)[0, 0]
    # the same holds for electrons, as make_l1 calls it
    electrons = exact.apply(counts * gain, electrons=True)
    assert np.max(np.abs(linearity.apply(electrons / gain) - counts)[1:]) < 0.05


def test_inl_uniform_grid_matches_interp():
    """The indexed INL interpolation reproduces np.interp, including out of
    range and NaN inputs."""
    rng = np.random.default_rng(2)
    ncols = 256 - 2 * parameters.nborder
    inl = _jagged_inl(rng, scale=1)
    coeffs = np.zeros((2, 10, ncols), dtype='f4')
    coeffs[1] = 1
    linearity = nonlinearity.Nonlinearity(
        coeffs=coeffs, gain=1, integralnonlinearity=inl, reftype='linearity')
    assert linearity._inl_step == 1
    counts = rng.uniform(-100, 70000, coeffs.shape[1:])
    counts[0, :3] = [np.nan, 0, 65535]
    indexed = linearity.inl_correction(counts)
    linearity._inl_step = None
    np.testing.assert_allclose(indexed, linearity.inl_correction(counts),
                               rtol=0, atol=1e-10)
    assert np.isnan(indexed[0, 0])


@pytest.mark.parametrize('exact', [True, False])
def test_l1_l2_rate_with_inverse_of(exact):
    """A ramp simulated by make_l1 with the inverse linearity and fit by
    make_l2 with the linearity recovers the input rate only if the inverse is
    exact; the CRDS pairs are off by ~0.3-0.6% in slope near the pedestal."""
    from romanisim import l1, image
    rng = np.random.default_rng(3)
    ncols = 256 - 2 * parameters.nborder
    shape = (40, ncols)
    inl = _jagged_inl(rng)
    a, b = 1e-6, 1e-12
    lin_coeffs = np.zeros((4,) + shape, dtype='f4')
    lin_coeffs[1:] = np.array([1, a, b])[:, None, None]
    # series inverse with a 0.5% error in its linear term
    inv_coeffs = np.zeros((4,) + shape, dtype='f4')
    inv_coeffs[1:] = np.array([1.005, -a, 2 * a ** 2 - b])[:, None, None]
    gain = 1.8
    linearity = nonlinearity.Nonlinearity(
        coeffs=lin_coeffs, gain=gain, integralnonlinearity=inl,
        reftype='linearity', getdq=True)
    inv_linearity = nonlinearity.Nonlinearity(
        coeffs=inv_coeffs, gain=gain, integralnonlinearity=inl, getdq=True,
        reftype='inverselinearity', inverse_of=linearity if exact else None)

    # one read per resultant: make_l2 linearizes resultant means, which with a
    # jagged INL is itself slightly off for multi-read resultants
    read_pattern = [[i] for i in range(1, 15)]
    exptime = parameters.read_time * read_pattern[-1][-1]
    electrons = 20000
    counts = galsim.ImageF(np.full(shape, electrons, dtype='f4'))
    resultants, dq = l1.make_l1(
        counts, read_pattern, read_noise=0, pedestal=10000 * gain,
        pedestal_extra_noise=0, gain=gain, inv_linearity=inv_linearity,
        rng=galsim.UniformDeviate(1))
    slope, _, _ = image.make_l2(resultants, read_pattern, read_noise=5,
                                gain=gain, flat=1, linearity=linearity,
                                darkrate=0)
    ratio = np.mean(slope) * exptime * gain / electrons
    if exact:
        assert np.abs(ratio - 1) < 1e-4
    else:
        assert np.abs(ratio - 1) > 3e-3
