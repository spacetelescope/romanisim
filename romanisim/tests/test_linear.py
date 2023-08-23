"""Unit tests for linear module.

Routines tested:
- repair_coefficients
- evaluate_nl_polynomial
- apply
"""
import os
import pytest
import numpy as np
from astropy import units as u
from astropy import stats
import crds

import roman_datamodels
from romanisim import nonlinearity, parameters
from romanisim import log


tijlist = [
    [[1], [2, 3], [4, 5], [6, 7]],
    [[100], [101, 102, 103], [110]],
    [[1], [2], [3, 4, 5, 100]],
]

ma_table_list = [
    [[1, 10], [11, 1], [12, 10], [30, 1], [40, 5], [50, 100]],
    [[1, 1]],
    [[1, 1], [10, 1]],
]


def test_linear_apply():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.asfarray([1.0, 0.7, 3.0e-6, 5.0e-12])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    gain = 4.0 * u.electron / u.DN
    counts[0, 0] = counts[0, 99] = counts[99, 0] = counts[99, 99] = 11.0

    linearity = nonlinearity.NL(lin_coeffs, gain=gain)

    res2 = linearity.apply(counts, electrons=True)

    assert res2[0,0] == res2[0,99]
    assert res2[99, 0] == res2[99, 99]
    assert res2[0, 0] == 2 * res2[99, 99]


def test_repair_coeffs():
    counts = np.random.poisson(100, size=(100, 100))

    coeffs = np.asfarray([0, 0.994, 3.0e-5, 5.0e-10, 7.0e-15])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0

    # Assign invalid coefficients to be repaired (no correction applied to pixels).
    lin_coeffs[:, 1, 1] *= 0
    lin_coeffs[2, 22, 22] = np.nan

    gain = 4.0 * u.electron / u.DN

    linearity = nonlinearity.NL(lin_coeffs, dq, gain=gain)

    assert linearity.dq[1, 1] == parameters.dqbits['nonlinear']
    assert linearity.dq[22, 22] == parameters.dqbits['nonlinear']
    # All other entries should be zero
    assert np.count_nonzero(linearity.dq) == 2

    res = linearity.apply(counts)

    assert res[1, 1] == counts[1, 1]
    assert res[22, 22] == counts[22, 22]
    # All other entries should be the same
    assert np.sum(res != counts) == np.prod(counts.shape) - 2

def test_electrons():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.asfarray([1.0, 0.7, 3.0e-6, 5.0e-12])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    gain = 4.0 * u.electron / u.DN

    linearity = nonlinearity.NL(lin_coeffs, gain=gain)

    res = linearity.apply(counts)

    res_elec = linearity.apply(gain * counts, electrons=True)

    assert np.all(res_elec[:] == gain * res[:])
    assert res_elec.unit == u.electron / u.DN
    assert not hasattr(res, "unit")

def test_reverse():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.asfarray([1.0, 0.7, 3.0e-6, 5.0e-12])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))

    lin_coeffs[:, 0:50, :] *= 2.0
    rev_lin_coeffs = lin_coeffs[::-1, ...]
    gain = 4.0 * u.electron / u.DN

    linearity = nonlinearity.NL(lin_coeffs, gain=gain)
    rev_linearity = nonlinearity.NL(rev_lin_coeffs, gain=gain)

    res = linearity.apply(counts)
    res_rev = rev_linearity.apply(counts, reversed=True)

    assert np.all(res_rev[:] == res[:])

@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason=(
        "Roman CRDS servers are not currently available outside the internal network"
    ),
)
def test_inverse_then_linearity():
    # Test that applying inverse linearity and then linearity returns the results to
    # the original value

    reffiles = crds.getreferences(
        {'roman.meta.instrument.name': 'WFI',
         'roman.meta.instrument.detector':'WFI01',
         'roman.meta.exposure.start_time': '2026-01-01T00:00:00'},
        reftypes=['inverselinearity', 'linearity'],
        observatory='roman')

    inverse_linearity_model = roman_datamodels.datamodels.InverselinearityRefModel(
        reffiles['inverselinearity'])
    linearity_model = roman_datamodels.datamodels.LinearityRefModel(
        reffiles['linearity'])


    inverse_linearity = nonlinearity.NL(
        inverse_linearity_model.coeffs[:,4:-4,4:-4], gain=1.0)
    linearity = nonlinearity.NL(
        linearity_model.coeffs[:,4:-4,4:-4], gain=1.0)

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

    assert rms < 1e-5
