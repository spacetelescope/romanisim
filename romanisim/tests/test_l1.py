"""Unit tests for L1 module.

Routines tested:
- validate_times
- tij_to_pij
- apportion_counts_to_resultants
- add_read_noise_to_resultants
- make_asdf
- read_pattern_to_tij
- make_l1
"""

import pytest
import numpy as np
from romanisim import l1, log
from romanisim.models import parameters, nonlinearity
import galsim
import asdf
import os


tijlist = [
    [[1], [2, 3], [4, 5], [6, 7]],
    [[100], [101, 102, 103], [110]],
    [[1], [2], [3, 4, 5, 100]],
]

read_pattern_list = [
    [[1 + x for x in range(10)], [11], [12 + x for x in range(10)], [30],
     [40 + x for x in range(5)], [50 + x for x in range(100)]],
    [[1]],
    [[1], [10]],
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
    pedestal = 1000  # pedestal in electrons
    pedestal_extra_noise = 20  # pedestal noise in electrons
    res1out = []
    res2out = []
    res3out = []
    res4out = []
    for tij in tijlist:
        # Test without pedestal
        rng1 = galsim.UniformDeviate(42)
        resultants, dq = l1.apportion_counts_to_resultants(counts, tij, rng=rng1)
        assert np.all(np.diff(resultants, axis=0) >= 0)
        assert np.all(resultants >= 0)
        assert np.all(resultants <= counts[None, :, :])
        res2 = l1.add_read_noise_to_resultants(resultants.copy(),  # DN
                                               tij)
        res3 = l1.add_read_noise_to_resultants(resultants.copy(), tij,
                                               read_noise=read_noise)
        assert np.all(res2 != resultants)
        assert np.all(res3 != resultants)

        # Test with pedestal but no pedestal noise (use same RNG seed for counts)
        rng2 = galsim.UniformDeviate(42)
        resultants_with_ped, dq2 = l1.apportion_counts_to_resultants(
            counts, tij, pedestal=pedestal, rng=rng2)
        # Should be offset by pedestal (before any linearity applied)
        assert np.all(resultants_with_ped >= pedestal)
        assert np.all(np.diff(resultants_with_ped, axis=0) >= 0)

        # Test with pedestal and pedestal noise
        rng3 = galsim.UniformDeviate(42)
        resultants_with_ped_noise, dq3 = l1.apportion_counts_to_resultants(
            counts, tij, pedestal=pedestal, pedestal_extra_noise=pedestal_extra_noise,
            rng=rng3)

        res1out.append(resultants)
        res2out.append(res2)
        res3out.append(res3)
        res4out.append(resultants_with_ped_noise)

        for restij, plane_index in zip(tij, np.arange(res3.shape[0])):
            predcounts = (np.mean(restij) * counts_no_poisson_noise
                          / tij[-1][-1])
            assert np.all((predcounts - resultants[plane_index])
                          < 10 * np.sqrt(predcounts))
            # agree with expected counts at 10 sigma.
            sdev = np.std(res3[plane_index] - resultants[plane_index])
            assert (np.abs(sdev - read_noise / np.sqrt(len(restij)))
                    < 20 * sdev / np.sqrt(2 * len(counts.ravel())))
            # Test pedestal noise: difference between resultants with/without pedestal noise
            sdev = np.std(resultants_with_ped_noise[plane_index]
                         - resultants_with_ped[plane_index])
            assert (np.abs(sdev - pedestal_extra_noise)
                    < 20 * sdev / np.sqrt(2 * len(counts.ravel())))
        # pedestal extra noise should be correlated and cancel out of the
        # first difference (allowing for small floating point precision differences)
        assert np.allclose(np.diff(resultants_with_ped_noise - resultants_with_ped, axis=0),
                          0, atol=1e-3)
    log.info('DMS220: successfully added read noise to resultants.')
    log.info('DMS229: successfully generated ramp from counts.')
    log.info('DMS223: successfully added correlated noise associated '
             'with frame zero')

    artifactdir = os.environ.get('TEST_ARTIFACT_DIR', None)
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'resultants': res1out,
                   'counts': counts,
                   'counts_no_poisson_noise': 100,
                   'tijlist': tijlist}
        af.write_to(os.path.join(artifactdir, 'dms229.asdf'))

        af = asdf.AsdfFile()
        af.tree = {'resultants': res4out,
                   'resultants_nonoise': res1out,
                   'counts': counts,
                   'counts_no_poisson_noise': 100,
                   'tijlist': tijlist}
        af.write_to(os.path.join(artifactdir, 'dms223.asdf'))


@pytest.mark.soctests
def test_linearized_counts_to_resultants():
    """Test that we can apportion linearized counts to resultants.
    Demonstrates DMS222: nonlinearity
    """
    counts = np.random.poisson(100, size=(100, 100))

    coeffs = np.array([0, 0.5, 0, 0, 0], dtype='f4')
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))

    rng1 = galsim.UniformDeviate(42)
    rng2 = galsim.UniformDeviate(42)

    # Create one bad coefficient
    lin_coeffs[1, -1, -1] = 0

    inv_linearity = nonlinearity.Nonlinearity(coeffs=lin_coeffs, getdq=True)

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
        assert np.all(dq[:, :-1, :-1] == dq2[:, :-1, :-1])
        assert np.all(dq2[:, -1, -1] == (dq[:, -1, -1]
                                         | parameters.dqbits['no_lin_corr']))
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


def test_read_pattern_to_tij():
    tij = l1.read_pattern_to_tij(4)
    assert l1.validate_times(tij)
    for read_pattern in read_pattern_list:
        tij = l1.read_pattern_to_tij(read_pattern)
        assert l1.validate_times(tij)

    # the reference read is prepended one read_time before the first read,
    # for any indexing convention, and leaves the science reads unchanged
    for read_pattern in read_pattern_list + [[[0], [1, 2], [3]]]:
        base = l1.read_pattern_to_tij(read_pattern)
        withref = l1.read_pattern_to_tij(read_pattern, reference_read=True)
        assert len(withref) == len(base) + 1
        assert len(withref[0]) == 1  # single read
        assert withref[0][0] == base[0][0] - parameters.read_time
        assert all(np.array_equal(a, b) for a, b in zip(withref[1:], base))
        assert l1.validate_times(withref)


@pytest.mark.soctests
def test_make_l1_and_asdf(tmp_path):
    """Make an L1 file and save it appropriately.
    Demonstrates DMS227.
    """
    # these two functions basically just wrap the above and we'll really
    # just test for sanity.
    counts = np.random.poisson(100, size=(100, 100))
    parameters.n_pix = 100
    for read_pattern in read_pattern_list:
        resultants, dq = l1.make_l1(galsim.Image(counts), read_pattern,
                                    gain=1)  # electron/DN
        assert resultants.shape[0] == len(read_pattern)
        assert resultants.shape[1] == counts.shape[0]
        assert resultants.shape[2] == counts.shape[1]
        # these contain read noise and shot noise, so it's not
        # clear what else to do with them?
        assert np.all(resultants == resultants.astype('i4'))
        # we could look for non-zero correlations from the IPC to
        # check that that is working?  But that is slightly annoying.
        resultants, dq = l1.make_l1(galsim.Image(counts), read_pattern,
                                    read_noise=0,  # DN
                                    pedestal_extra_noise=0,  # electron
                                    gain=1)  # electron/DN
        # technically, resultants are in DN and pedestal is in electrons,
        # but in this test the gain is one and so we ignore this distinction.
        assert np.all(resultants - parameters.pedestal
                      <= np.max(counts[None, ...]))
        # because of IPC, one can't require that each pixel is smaller
        # than the number of counts
        assert np.all(resultants >= 0)  # DN
        assert np.all(np.diff(resultants, axis=0) >= 0)  # DN
        res_forasdf, extras = l1.make_asdf(
            resultants, filepath=tmp_path / 'tmp.asdf')
        af = asdf.AsdfFile()
        af.tree = {'roman': res_forasdf}
        af.validate()
        resultants, dq = l1.make_l1(galsim.Image(np.full((100, 100), 10**7)),
                                    read_pattern, gain=1,  # electron/DN
                                    saturation=10**6)  # DN
        assert np.all((dq[-1] & parameters.dqbits['saturated']) != 0)
        resultants, dq = l1.make_l1(galsim.Image(np.zeros((100, 100))),
                                    read_pattern, gain=1,  # electron/DN
                                    read_noise=0,  # DN
                                    pedestal_extra_noise=0,  # electron
                                    crparam=dict())
        # technically, resultants are in DN and pedestal is in electrons,
        # but in this test the gain is one and so we ignore this distinction.
        assert np.all((resultants[0] - parameters.pedestal == 0)
                      | ((dq[0] & parameters.dqbits['jump_det']) != 0))
    log.info('DMS227: successfully made an L1 file that validates.')


def test_encode_decode_reference_read():
    """Encoding resultants against a reference read is invertible."""
    res = np.round(np.random.uniform(4000, 20000, size=(5, 20, 20)))
    ref = np.round(np.random.uniform(4900, 5100, size=(20, 20)))

    enc = l1.encode_reference_read(res, ref, 4000)
    assert enc.dtype == np.uint16
    assert np.all(l1.decode_reference_read(enc, ref, 4000) == res)
    assert np.all(enc == res - ref[None] + 4000)

    # the offset exists to keep the encoded data off the zero rail; without it
    # any resultant below the reference read clips, and clipping is the only
    # thing that breaks the inversion
    enc0 = l1.encode_reference_read(res, ref, 0)
    clipped = res - ref[None] < 0
    assert np.any(clipped)
    dec0 = l1.decode_reference_read(enc0, ref, 0)
    assert np.all(dec0[~clipped] == res[~clipped])
    assert np.all(dec0[clipped] > res[clipped])

    # default offset comes from parameters
    assert np.all(l1.encode_reference_read(res, ref)
                  == l1.encode_reference_read(
                      res, ref, parameters.data_encoding_offset))


def test_reference_read(tmp_path):
    """Simulating a reference read and packaging it into an L1 file."""
    counts = np.random.poisson(100, size=(100, 100))
    parameters.n_pix = 100
    nb = parameters.nborder
    read_pattern = read_pattern_list[0]
    shape = (len(read_pattern),) + counts.shape
    kw = dict(pedestal_extra_noise=0, gain=1, crparam=None, seed=12)

    # with no read noise, turning on the reference read must not perturb the
    # resultants: it must not consume the shared RNG before they are built
    res0, dq0 = l1.make_l1(galsim.Image(counts), read_pattern,
                           read_noise=0, **kw)
    res1, dq1, ref1 = l1.make_l1(galsim.Image(counts), read_pattern,
                                 read_noise=0, reference_read=True, **kw)
    assert res1.shape == res0.shape == shape
    assert dq1.shape == dq0.shape == shape  # dq gets no reference read plane
    assert np.all(res1 == res0)
    assert np.all(ref1 == parameters.pedestal)  # linearity here is the identity

    # the reference read is a single read, so it carries the full read noise
    res1, dq1, ref1 = l1.make_l1(galsim.Image(counts), read_pattern,
                                 read_noise=5, reference_read=True, **kw)
    assert np.abs(np.std(ref1 - parameters.pedestal) / 5 - 1) < 0.2

    out, extras = l1.make_asdf(res1, dq=dq1, reference_read=ref1,
                               data_encoding_offset=4000,
                               filepath=tmp_path / 'refread.asdf')
    assert out['meta']['instrument']['data_encoding_offset'] == 4000
    assert np.all(out['reference_read'][nb:-nb, nb:-nb] == ref1)
    assert np.all(l1.decode_reference_read(
        out['data'][:, nb:-nb, nb:-nb], out['reference_read'][nb:-nb, nb:-nb],
        4000) == res1)
    # amp33 is not simulated, but must survive the round trip rather than
    # underflowing the unsigned data
    assert np.all(l1.decode_reference_read(
        out['amp33'], out['reference_amp33'], 4000) == 0)
    af = asdf.AsdfFile()
    af.tree = {'roman': out}
    af.validate()

    # no reference read requested -> none of the extra structure appears
    out, extras = l1.make_asdf(res0, dq=dq0)
    assert 'reference_read' not in out
    assert 'data_encoding_offset' not in out['meta']['instrument']
