"""Unit tests for CR library."""

from romanisim import cr
import numpy as np


def test_traverse():
    ii, jj, lengths = cr.traverse([53.6, 77.1], [54.8, 76.1])
    # set of pixels is unique
    assert len(set(zip(ii, jj))) == len(ii)
    i1, j1 = np.random.random((2, 100)) * 100
    i2, j2 = np.random.random((2, 100)) * 100
    for i in range(len(i1)):
        ii, jj, lengths = cr.traverse((i1[i], j1[i]), (i2[i], j2[i]), 100, 100)
        totlen = np.hypot(i1[i] - i2[i], j1[i] - j2[i])
        assert len(set(zip(ii, jj))) == len(ii)
        assert np.all(lengths > 0)
        assert np.all((lengths < totlen) | np.isclose(lengths, totlen))


def test_create_sampler():
    xx = np.linspace(0, 1, 1000)
    sampler = cr.create_sampler(lambda x: np.ones_like(x), xx)
    assert np.allclose(sampler(xx), xx)


def test_moyal():
    xx = np.linspace(-100, 100, 10000)
    moyal = cr.moyal_distribution(xx, 0, 1)
    # what can I test here??
    # at least that it's positive and obeys the right scaling symmetry...
    assert np.all(moyal >= 0)
    assert np.isclose(cr.moyal_distribution(10, 0, 1),
                      cr.moyal_distribution(0, -100, 10))


def test_power_law():
    xx = np.linspace(-10, 10, 1000)
    p1 = 1
    p2 = 2
    pl1 = cr.power_law_distribution(xx, p1)
    pl2 = cr.power_law_distribution(xx, p2)
    assert pl1[-1] / pl1[0] < pl2[-1] / pl2[0]
    assert np.isclose(pl1[-1] / pl1[0], (xx[-1] / xx[0])**p1)
    assert np.isclose(pl2[-1] / pl2[0], (xx[-1] / xx[0])**p2)


def test_sample_cr_params():
    x, y, phi, ll, dEdx = cr.sample_cr_params(
        10000, N_i=100, N_j=100, min_dEdx=10, max_dEdx=100,
        min_cr_len=20, max_cr_len=40)
    # umm, test some simple bounds
    assert (np.max(x) < 100.5) & (np.min(x) > -0.5)
    assert (np.max(y) < 100.5) & (np.min(y) > -0.5)
    assert (np.min(phi) >= 0) & (np.max(phi) < 2 * np.pi)
    assert (np.max(ll) < 40) & (np.min(ll) > 20)
    assert (np.max(dEdx) < 100) & (np.min(dEdx) > 10)


def test_simulate_crs():
    im = np.zeros((100, 100), dtype='i4')
    cr.simulate_crs(im, 10)
    assert np.all(im >= 0)
    assert np.sum(im) >= 0
