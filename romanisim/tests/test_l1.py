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

import numpy as np
from romanisim import l1
import galsim
import galsim.roman
import asdf

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
            pij, np.diff(np.concatenate(tij), prepend=0)/tij[-1][-1])
        


def test_apportion_counts_to_resultants():
    # we'll skip the linearity tests until new linearity files with
    # inverse coefficients are available in CRDS.
    counts = np.random.poisson(100, size=(100, 100))
    read_noise = 10
    for tij in tijlist:
        resultants = l1.apportion_counts_to_resultants(counts, tij)
        assert np.all(np.diff(resultants, axis=0) >= 0)
        assert np.all(resultants >= 0)
        assert np.all(resultants <= counts[None, :, :])
        res2 = l1.add_read_noise_to_resultants(resultants.copy(), tij)
        res3 = l1.add_read_noise_to_resultants(resultants.copy(), tij,
                                               read_noise=read_noise)
        assert np.all(res2 != resultants)
        assert np.all(res3 != resultants)
        for restij, plane_index in zip(tij, np.arange(res3.shape[0])):
            sdev = np.std(res3[plane_index] - resultants[plane_index])
            assert (sdev - read_noise / np.sqrt(len(restij))
                    < 20 * sdev / np.sqrt(2 * len(counts.ravel())))
    
def test_ma_table_to_tij():
    tij = l1.ma_table_to_tij(1)
    # this is the only numbered ma_table that we have presently provided.
    assert l1.validate_times(tij)
    for ma_table in ma_table_list:
        tij = l1.ma_table_to_tij(ma_table)
        assert l1.validate_times(tij)
    

def test_make_l1_and_asdf(tmp_path):
    # these two functions basically just wrap the above and we'll really
    # just test for sanity.
    counts = np.random.poisson(100, size=(100, 100))
    galsim.roman.n_pix = 100
    for ma_table in ma_table_list:
        resultants = l1.make_l1(galsim.Image(counts), ma_table, gain=1)
        assert resultants.shape[0] == len(ma_table)
        assert resultants.shape[1] == counts.shape[0]
        assert resultants.shape[2] == counts.shape[1]
        # these contain read noise and shot noise, so it's not
        # clear what else to do with them?
        assert np.all(resultants == resultants.astype('i4'))
        # we could look for non-zero correlations from the IPC to
        # check that that is working?  But that is slightly annoying.
        resultants = l1.make_l1(galsim.Image(counts), ma_table, read_noise=0)
        assert np.all(resultants <= np.max(counts[None, ...]))
        # because of IPC, one can't require that each pixel is smaller
        # than the number of counts
        assert np.all(resultants >= 0)
        assert np.all(np.diff(resultants, axis=0) >= 0)
        res_forasdf = l1.make_asdf(resultants, filepath=tmp_path / 'tmp.asdf')
        af = asdf.AsdfFile()
        af.tree = {'roman': res_forasdf}
        af.validate()
