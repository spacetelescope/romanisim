"""Unit tests for linear module.

Routines tested:
- repair_coefficients
- evaluate_nl_polynomial
- apply
"""
import numpy as np
from astropy import stats
import crds

import roman_datamodels
from romanisim import nonlinearity, parameters
from romanisim import log


def test_linear_apply():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.array([0, 0.994, 3.0e-5, 5.0e-10, 7.0e-15], dtype='f4')
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    gain = 4.0  # electron/DN
    counts[0, 0] = counts[0, 99] = counts[99, 0] = counts[99, 99] = 11.0

    linearity = nonlinearity.NL(lin_coeffs, gain=gain)

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

    linearity = nonlinearity.NL(lin_coeffs, gain=gain)

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

    linearity = nonlinearity.NL(lin_coeffs, gain=gain)

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

    linearity = nonlinearity.NL(lin_coeffs, gain=gain)
    rev_linearity = nonlinearity.NL(rev_lin_coeffs, gain=gain)

    res = linearity.apply(counts)
    res_rev = rev_linearity.apply(counts, reversed=True)

    assert np.all(res_rev[:] == res[:])


def test_inl_correction():
    """Test integral nonlinearity correction with a simple +1 correction."""
    from types import SimpleNamespace

    # Create a mock INL model with all corrections = 1
    ncols = 256  # 2 channels
    inl_model = SimpleNamespace(
        value=np.arange(65536, dtype='f4'),
        inl_table=SimpleNamespace()
    )
    for start_col in range(0, ncols, 128):
        channel_num = start_col // 128 + 1
        attr_name = f"science_channel_{channel_num:02d}"
        setattr(inl_model.inl_table, attr_name,
                SimpleNamespace(correction=np.ones(65536, dtype='f4')))

    # Identity polynomial coefficients (output = input)
    identity_coeffs = np.zeros((5, 100, ncols), dtype='f4')
    identity_coeffs[1, :, :] = 1.0  # linear term = 1

    # Test forward (linearity) adds +1
    linearity = nonlinearity.NL(
        identity_coeffs, gain=1.0, integralnonlinearity=inl_model, inverse=False
    )
    counts = np.ones((100, ncols), dtype='f4') * 1000
    result = linearity.apply(counts)
    np.testing.assert_allclose(result, counts + 1)

    # Test inverse subtracts 1 (correction is negated)
    inv_linearity = nonlinearity.NL(
        identity_coeffs, gain=1.0, integralnonlinearity=inl_model, inverse=True
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

    inverse_linearity = nonlinearity.NL(
        inverse_linearity_model.coeffs[:, 4:-4, 4:-4], gain=1.0)
    linearity = nonlinearity.NL(
        linearity_model.coeffs[:, 4:-4, 4:-4], gain=1.0)

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
