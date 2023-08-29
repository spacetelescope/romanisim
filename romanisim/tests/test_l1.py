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
from romanisim import l1, log, parameters, nonlinearity
import galsim
import galsim.roman
import asdf
from astropy import units as u
import os


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
    Demonstrates DMS220, 229.
    """
    # we'll skip the linearity tests until new linearity files with
    # inverse coefficients are available in CRDS.
    counts_no_poisson_noise = 100
    counts = np.random.poisson(counts_no_poisson_noise, size=(100, 100))
    read_noise = 10
    res1out = []
    res2out = []
    res3out = []
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
        res1out.append(resultants)
        res2out.append(res2)
        res3out.append(res3)
        for restij, plane_index in zip(tij, np.arange(res3.shape[0])):
            predcounts = (np.mean(restij) * counts_no_poisson_noise
                          / tij[-1][-1])
            assert np.all((predcounts - resultants[plane_index])
                          < 10 * np.sqrt(predcounts))
            # agree with expected counts at 10 sigma.
            sdev = np.std(res3[plane_index] - resultants[plane_index])
            assert (np.abs(sdev - read_noise / np.sqrt(len(restij)))
                    < 20 * sdev / np.sqrt(2 * len(counts.ravel())))
    log.info('DMS220: successfully added read noise to resultants.')
    log.info('DMS229: successfully generated ramp from counts.')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'resultants': res1out,
                   'counts': counts,
                   'counts_no_poisson_noise': 100,
                   'tijlist': tijlist}
        af.write_to(os.path.join(artifactdir, 'dms229.asdf'))


@pytest.mark.soctests
def test_linearized_counts_to_resultants():
    """Test that we can apportion linearized counts to resultants.
    """
    counts = np.random.poisson(100, size=(100,100))

    coeffs = np.asfarray([0, 0.5, 0, 0, 0])
    lin_coeffs = np.tile(coeffs[:,np.newaxis,np.newaxis], (1, 100, 100))

    rng1 = galsim.UniformDeviate(42)
    rng2 = galsim.UniformDeviate(42)

    # Create one bad coefficient
    lin_coeffs[1,-1,-1] = 0

    inv_linearity = nonlinearity.NL(lin_coeffs)

    for tij in tijlist:
        resultants, dq = l1.apportion_counts_to_resultants(
            counts, tij, rng=rng1)

        res2, dq2 = l1.apportion_counts_to_resultants(
            counts, tij, rng=rng2, inv_linearity=inv_linearity)

        # Ensure that the linear coefficients were actually applied
        assert np.all(res2 >= 0)
        assert np.any(res2 < resultants)
        assert np.all(res2 <= resultants)
        # Test that the median difference between the original data and the
        # data after applying the nonlinearity is two; i.e., the value we
        # put into the nonlinearity correction polynomial.
        medratio = np.median(resultants[res2 != 0] / res2[res2 != 0])
        assert np.isclose(medratio, 2.0, atol=1e-6)
        # also test that correctly propagate the nonlinearity DQ bits.
        assert np.all(dq[:,:-1,:-1] == dq2[:,:-1,:-1])
        assert np.all(dq2[:,-1,-1] == np.bitwise_or(dq[:,-1,-1], parameters.dqbits['nonlinear']))
    log.info('DMS222: successfully applied nonlinearity to resultants.')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'resultants': resultants,
                   'dq': dq,
                   'resultants-with-nonlinearity': res2,
                   'dq-with-nonlinearity': dq2,
                   'coeffs': coeffs}
        af.write_to(os.path.join(artifactdir, 'dms222.asdf'))




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
    # checked that: we only added photons; we didn't touch anywhere where we
    # didn't inject photons; the flux in the ramp to which we injected
    # a source matches the injected flux to within 10 sigma.
    # factor of mean(tij[-1] / tij[-1][-1]) accounts for the fact that the
    # total number of counts in the last read is larger than the
    # number recorded in the last resultant, since the last resultant averages
    # over several reads.
    log.info('DMS225: successfully injected a source into a ramp.')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'originalramp': ramp,
                   'newramp': newramp,
                   'flux': flux,
                   'tij': tij}
        af.write_to(os.path.join(artifactdir, 'dms225.asdf'))



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
        res_forasdf, extras = l1.make_asdf(
            resultants, filepath=tmp_path / 'tmp.asdf')
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
