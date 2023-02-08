"""Unit tests for CR library."""

from romanisim import cr


def test_traverse():
    ii, jj, lengths = cr.traverse([53.6, 77.1], [54.8, 76.1])
    # set of pixels is unique
    assert len(set(zip(ii, jj))) == len(ii)
