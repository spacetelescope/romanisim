"""
Regression tests for Level 1 generation.
"""

import pytest

@pytest.mark.bigdata
def test_dummy_reg():
    assert 1>0

@pytest.mark.bigdata
@pytest.mark.soctests
def test_dummy_soctest_reg():
    assert 1>0
