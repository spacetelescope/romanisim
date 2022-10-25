"""
Unit tests for utility functions.
"""

import pytest

def test_dummy():
    assert 1>0

@pytest.mark.soctests
def test_dummy_soctest():
    assert 1>0
