"""Test ris_make_utils module.
Routines tested:
* set_metadata
* create_catalog
* simulate_image_file
"""

import types
import pytest
import numpy as np
from romanisim import ris_make_utils
import asdf
import galsim


def test_set_metadata():
    meta = ris_make_utils.set_metadata()
    assert meta is not None
    assert len(meta) > 0
    ris_make_utils.set_metadata(meta, bandpass='F158')
    assert meta['instrument']['optical_element'] == 'F158'


def test_create_catalog(tmp_path):
    with pytest.raises(ValueError):
        cat = ris_make_utils.create_catalog()
    from astropy.table import Table
    tabpath = tmp_path / 'table.ecsv'
    tab = Table(np.zeros(1, dtype=[('ra', 'f4')]))
    tab.write(tabpath)
    cat = ris_make_utils.create_catalog(catalog_name=tabpath, usecrds=False)
    assert len(cat) == 1
    cat = ris_make_utils.create_catalog(metadata=ris_make_utils.set_metadata(),
                                        nobj=1000, usecrds=False)
    assert len(cat) == 1000


def test_simulate_image_file(tmp_path):
    args = types.SimpleNamespace()
    meta = ris_make_utils.set_metadata()
    cat = ris_make_utils.create_catalog(meta, nobj=1, usecrds=False)
    args.filename = tmp_path / 'im.asdf'
    args.usecrds = False
    args.webbpsf = False
    args.level = 0
    galsim.roman.n_pix = 100
    ris_make_utils.simulate_image_file(args, meta, cat)
    im = asdf.open(args.filename)
    assert im['roman']['data'].shape == (100, 100)
    # we made an image


def test_parse_filename():
    assert ris_make_utils.parse_filename('blah') is None
    obs = ris_make_utils.parse_filename(
        'r9999901001001001001_01101_0001_uncal.asdf')
    assert obs is not None
    assert obs['program'] == '99999'
    assert obs['pass'] == 1
