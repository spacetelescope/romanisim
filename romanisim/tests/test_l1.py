"""Unit tests for L1 module.

Routines tested:
- validate_times
- tij_to_pij
- apportion_counts_to_resultants
- add_read_noise_to_resultants
- make_asdf
- ma_table_to_tij
- make_l1
"""

import pytest
import numpy as np
from romanisim import l1, log, parameters
import galsim
import galsim.roman
import asdf
from astropy import units as u


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


def test_validate_times():
    assert l1.validate_times([[0, 1], [2, 3, 4], [5, 6, 7, 8], [9], [10]])
    assert l1.validate_times([[-1, 0], [10, 11], [12, 20], [100]])
    assert not l1.validate_times([[0, 0], [1, 2], [3, 4]])
    assert not l1.validate_times([[0, 1], [1, 2], [3, 4]])
    assert not l1.validate_times([[0, -1], [1, 2], [3, 4]])


def test_tij_to_pij():
    for tij in tijlist:
        pij = l1.tij_to_pij(tij, remaining=True)
        pij = np.concatenate(pij)
        assert pij[-1] == 1
        assert np.all(pij > 0)
        assert np.all(pij <= 1)
        pij = l1.tij_to_pij(tij)
        pij = np.concatenate(pij)
        assert np.all(pij > 0)
        assert np.all(pij <= 1)
        assert np.allclose(
            pij, np.diff(np.concatenate(tij), prepend=0) / tij[-1][-1])


@pytest.mark.soctests
def test_apportion_counts_to_resultants():
    """Test that we can apportion counts to resultants and appropriately add
    read noise to those resultants, fulfilling DMS220.
    """
    # we'll skip the linearity tests until new linearity files with
    # inverse coefficients are available in CRDS.
    counts = np.random.poisson(100, size=(100, 100))
    read_noise = 10
    for tij in tijlist:
        resultants, dq = l1.apportion_counts_to_resultants(counts, tij)
        assert np.all(np.diff(resultants, axis=0) >= 0)
        assert np.all(resultants >= 0)
        assert np.all(resultants <= counts[None, :, :])
        res2 = l1.add_read_noise_to_resultants(resultants.copy() * u.DN,
                                               tij)
        res3 = l1.add_read_noise_to_resultants(resultants.copy(), tij,
                                               read_noise=read_noise)
        assert np.all(res2 != resultants)
        assert np.all(res3 != resultants)
        for restij, plane_index in zip(tij, np.arange(res3.shape[0])):
            sdev = np.std(res3[plane_index] - resultants[plane_index])
            assert (np.abs(sdev - read_noise / np.sqrt(len(restij)))
                    < 20 * sdev / np.sqrt(2 * len(counts.ravel())))
    log.info('DMS220: successfully added read noise to resultants.')


@pytest.mark.soctests
def test_inject_source_into_ramp():
    """Inject a source into a ramp.
    Demonstrates DMS225.
    """
    ramp = np.zeros((6, 100, 100), dtype='f4')
    sourcecounts = np.zeros((100, 100), dtype='f4')
    flux = 10000
    sourcecounts[49, 49] = flux  # delta function source with 100 counts.
    tij = np.arange(1, 6 * 3 + 1).reshape(6, 3)
    resultants, dq = l1.apportion_counts_to_resultants(sourcecounts, tij)
    newramp = ramp + resultants
    assert np.all(newramp >= ramp)
    # only added photons
    assert np.sum(newramp == ramp) == (100 * 100 - 1) * 6
    # only added to the one pixel
    injectedsource = (newramp - ramp)[:, 49, 49]
    assert (np.abs(injectedsource[-1] - flux * np.mean(tij[-1]) / tij[-1][-1])
            < 10 * np.sqrt(flux))
    # added correct number of photons
    log.info('DMS225: successfully injected a source into a ramp.')


@pytest.mark.soctests
def test_ipc():
    """Convolve an image with an IPC kernel.
    Demonstrates DMS226.
    """
    nresultant = 6
    testim = np.zeros((nresultant, 101, 101), dtype='f4')
    flux = 1e6
    testim[:, 50, 50] = flux
    kernel = np.ones((3, 3), dtype='f4')
    kernel /= np.sum(kernel)
    newim = l1.add_ipc(testim, kernel)
    assert np.sum(~np.isclose(newim, testim)) == 9 * nresultant
    assert (np.sum(np.isclose(newim[:, 49:52, 49:52], flux / 9))
            == 9 * nresultant)
    log.info('DMS226: successfully convolved image with IPC kernel.')


def test_ma_table_to_tij():
    tij = l1.ma_table_to_tij(1)
    # this is the only numbered ma_table that we have presently provided.
    assert l1.validate_times(tij)
    for ma_table in ma_table_list:
        tij = l1.ma_table_to_tij(ma_table)
        assert l1.validate_times(tij)


@pytest.mark.soctests
def test_make_l1_and_asdf(tmp_path):
    """Make an L1 file and save it appropriately.
    Demonstrates DMS227.
    """
    # these two functions basically just wrap the above and we'll really
    # just test for sanity.
    counts = np.random.poisson(100, size=(100, 100))
    galsim.roman.n_pix = 100
    for ma_table in ma_table_list:
        resultants, dq = l1.make_l1(galsim.Image(counts), ma_table,
                                    gain=1 * u.electron / u.DN)
        assert resultants.shape[0] == len(ma_table)
        assert resultants.shape[1] == counts.shape[0]
        assert resultants.shape[2] == counts.shape[1]
        # these contain read noise and shot noise, so it's not
        # clear what else to do with them?
        assert np.all(resultants == resultants.astype('i4'))
        # we could look for non-zero correlations from the IPC to
        # check that that is working?  But that is slightly annoying.
        resultants, dq = l1.make_l1(galsim.Image(counts), ma_table,
                                    read_noise=0 * u.DN,
                                    gain=1 * u.electron / u.DN)
        assert np.all(resultants - parameters.pedestal
                      <= np.max(counts[None, ...] * u.DN))
        # because of IPC, one can't require that each pixel is smaller
        # than the number of counts
        assert np.all(resultants >= 0 * u.DN)
        assert np.all(np.diff(resultants, axis=0) >= 0 * u.DN)
        res_forasdf = l1.make_asdf(resultants, filepath=tmp_path / 'tmp.asdf')
        af = asdf.AsdfFile()
        af.tree = {'roman': res_forasdf}
        af.validate()
        resultants, dq = l1.make_l1(galsim.Image(np.full((100, 100), 10**7)),
                                    ma_table, gain=1 * u.electron / u.DN,
                                    saturation=10**6 * u.DN)
        assert np.all((dq[-1] & parameters.dqbits['saturated']) != 0)
        resultants, dq = l1.make_l1(galsim.Image(np.zeros((100, 100))),
                                    ma_table, gain=1 * u.electron / u.DN,
                                    read_noise=0 * u.DN, crparam=dict())
        assert np.all((resultants[0] - parameters.pedestal == 0)
                      | ((dq[0] & parameters.dqbits['jump_det']) != 0))
    log.info('DMS227: successfully made an L1 file that validates.')
