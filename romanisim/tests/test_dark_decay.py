"""Unit tests for dark decay simulation.

Routines tested:
- frame_read_times
- dark_decay_for_read
- apply_dark_decay
- make_l1 / make_l2 round-trip
"""
import numpy as np
import galsim

from romanisim import l1, parameters
from romanisim.image import make_l2


def test_frame_read_times_shape():
    """Test that frame_read_times returns the right shape."""
    rt = l1.frame_read_times(3.0, 7)
    assert rt.shape == (4096, 4096)
    rt_trimmed = l1.frame_read_times(3.0, 7, nborder=4)
    assert rt_trimmed.shape == (4088, 4088)


def test_frame_read_times_range():
    """Test that pixel read times span one frame_time."""
    frame_time = 3.0
    rt = l1.frame_read_times(frame_time, 7, t0=0)
    assert rt.min() >= 0
    assert rt.max() < frame_time


def test_frame_read_times_t0():
    """Test that t0 shifts all read times."""
    rt0 = l1.frame_read_times(3.0, 7, t0=0)
    rt10 = l1.frame_read_times(3.0, 7, t0=10)
    np.testing.assert_allclose(rt10, rt0 + 10)


def test_frame_read_times_sca_flip():
    """Test that SCA % 3 == 0 flips columns, others flip rows."""
    frame_time = 3.0
    # SCA 3 (3 % 3 == 0): column flip
    rt3 = l1.frame_read_times(frame_time, 3, t0=0)
    # SCA 7 (7 % 3 != 0): row flip
    rt7 = l1.frame_read_times(frame_time, 7, t0=0)
    # Both should have the same set of values, just rearranged
    assert np.isclose(rt3.min(), rt7.min())
    assert np.isclose(rt3.max(), rt7.max())
    # But the arrays should differ
    assert not np.allclose(rt3, rt7)


def test_dark_decay_for_read_amplitude():
    """Test that the dark decay signal at early times is close to the
    amplitude."""
    darkdecaysignal = dict(amplitude=1.0, time_constant=23.0, sca=7)
    # First read at t = read_time; frame_offset = 1.5 means amplitude
    # is referenced to t = 1.5 * read_time.  For the first read at
    # t ~ read_time, the signal should be close to amplitude since
    # read_time << time_constant.
    signal = l1.dark_decay_for_read(darkdecaysignal, parameters.read_time)
    assert np.abs(np.mean(signal) - 1.0) < 0.1


def test_dark_decay_for_read_decays():
    """Test that the dark decay signal decreases with time."""
    darkdecaysignal = dict(amplitude=1.0, time_constant=23.0, sca=7)
    signal_early = l1.dark_decay_for_read(
        darkdecaysignal, parameters.read_time)
    signal_late = l1.dark_decay_for_read(
        darkdecaysignal, 100 * parameters.read_time)
    assert np.mean(signal_early) > np.mean(signal_late)


def test_dark_decay_for_read_amplitude_scales():
    """Test that doubling amplitude doubles the signal."""
    dds1 = dict(amplitude=1.0, time_constant=23.0, sca=7)
    dds2 = dict(amplitude=2.0, time_constant=23.0, sca=7)
    t = 5 * parameters.read_time
    signal1 = l1.dark_decay_for_read(dds1, t)
    signal2 = l1.dark_decay_for_read(dds2, t)
    np.testing.assert_allclose(signal2, 2 * signal1)


def test_apply_dark_decay_adds_signal():
    """Test that apply_dark_decay with sign=+1 adds positive signal
    that decays over time."""
    read_pattern = [[1], [2], [3], [4], [5], [6]]
    nx, ny = 4088, 4088
    resultants = np.zeros((len(read_pattern), nx, ny), dtype='f4')
    darkdecaysignal = dict(amplitude=1.0, time_constant=23.0, sca=7)

    l1.apply_dark_decay(resultants, darkdecaysignal, read_pattern, sign=1)

    # All values should be positive
    assert np.all(resultants > 0)

    # Each resultant's mean signal should decay
    means = [np.mean(resultants[i]) for i in range(len(read_pattern))]
    for i in range(len(means) - 1):
        assert means[i] > means[i + 1]


def test_apply_dark_decay_roundtrip():
    """Test that adding then subtracting dark decay recovers original."""
    read_pattern = [[1], [2, 3], [4, 5, 6]]
    nx, ny = 4088, 4088
    original = np.ones((len(read_pattern), nx, ny), dtype='f4') * 100
    resultants = original.copy()
    darkdecaysignal = dict(amplitude=1.0, time_constant=23.0, sca=7)

    l1.apply_dark_decay(resultants, darkdecaysignal, read_pattern, sign=1)
    assert not np.allclose(resultants, original)

    l1.apply_dark_decay(resultants, darkdecaysignal, read_pattern, sign=-1)
    np.testing.assert_allclose(resultants, original, atol=1e-5)


def test_apply_dark_decay_multiread_dilution():
    """Test that averaging more reads in a resultant dilutes the signal,
    as in the romancal test."""
    nx, ny = 4088, 4088
    darkdecaysignal = dict(amplitude=1.0, time_constant=23.0, sca=7)

    # Resultant 0 is frame 1 only
    rp1 = [[1], [2, 3]]
    res1 = np.zeros((2, nx, ny), dtype='f4')
    l1.apply_dark_decay(res1, darkdecaysignal, rp1, sign=1)

    # Resultant 0 is average of frames 1 and 2
    rp2 = [[1, 2], [3]]
    res2 = np.zeros((2, nx, ny), dtype='f4')
    l1.apply_dark_decay(res2, darkdecaysignal, rp2, sign=1)

    # Single early read should have larger signal than average with
    # a later read
    assert np.all(np.abs(res1[0]) >= np.abs(res2[0]))


def test_apply_dark_decay_amplitude_match():
    """Test that the amplitude is about right in the first read.
    Frame time is much smaller than the time constant, so the
    amplitude should closely match the mean of the first resultant."""
    nx, ny = 4088, 4088
    darkdecaysignal = dict(amplitude=1.0, time_constant=23.0, sca=7)
    rp = [[1], [2, 3]]
    res = np.zeros((2, nx, ny), dtype='f4')
    l1.apply_dark_decay(res, darkdecaysignal, rp, sign=1)
    assert np.abs(np.mean(res[0]) - 1.0) < 0.01


def test_dark_decay_l1_l2_roundtrip():
    """Test that dark decay added in make_l1 is removed in make_l2,
    recovering a zero count rate from a zero count image."""
    read_pattern = [[1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
    # huge amplitude so that it's much larger than sources of noise
    # in the round trip.
    darkdecaysignal = dict(amplitude=1000.0, time_constant=3.0, sca=7)
    nx, ny = 4088, 4088
    counts = galsim.Image(np.zeros((nx, ny), dtype='f4'), xmin=0, ymin=0)

    resultants, dq = l1.make_l1(
        counts, read_pattern,
        read_noise=0, pedestal=0, pedestal_extra_noise=0,
        darkdecaysignal=darkdecaysignal, seed=42)

    slopes, readvar, poissonvar = make_l2(
        resultants, read_pattern,
        read_noise=0,
        darkdecaysignal=darkdecaysignal, dq=dq)

    # Quantization in make_l1 (rounding to integer DN) prevents exact
    # cancellation; check that residuals are small relative to the
    # amplitude.
    np.testing.assert_allclose(slopes, 0, atol=0.2)
