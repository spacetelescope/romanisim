"""Test ris_make_utils module.
Routines tested:
* set_metadata
* create_catalog
* simulate_image_file
* add_meta_args / apply_meta_args
"""

import types
import pytest
import numpy as np
from romanisim import ris_make_utils
from romanisim.models import parameters
import asdf


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
    args.psftype=None
    args.level = 0
    args.sca = 7
    args.bandpass = 'F184'
    args.pretend_spectral = None
    parameters.n_pix = 100
    ris_make_utils.simulate_image_file(args, meta, cat)
    im = asdf.open(args.filename)
    assert im['roman']['data'].shape == (100, 100)
    # we made an image


def test_parse_filename():
    assert ris_make_utils.parse_filename('blah') is None
    obs = ris_make_utils.parse_filename(
        'r9999901001001001001_01101_0001_uncal.asdf')
    assert obs is not None
    assert obs['program'] == 99999
    assert obs['pass'] == 1


def test_apply_meta_args():
    import argparse
    meta = ris_make_utils.set_metadata()

    # integer coercion and nested key creation
    args = argparse.Namespace(meta=['visit.nexposures=4'])
    ris_make_utils.apply_meta_args(args, meta)
    assert meta['visit']['nexposures'] == 4
    assert isinstance(meta['visit']['nexposures'], int)

    # string fallback and multiple overrides
    args = argparse.Namespace(meta=['visit.visit_type=GENERIC',
                                    'visit.nexposures=2'])
    ris_make_utils.apply_meta_args(args, meta)
    assert meta['visit']['visit_type'] == 'GENERIC'
    assert meta['visit']['nexposures'] == 2

    # None arg is a no-op
    original = meta['visit']['nexposures']
    args = argparse.Namespace(meta=None)
    ris_make_utils.apply_meta_args(args, meta)
    assert meta['visit']['nexposures'] == original


def test_format_filename():
    from romanisim.ris_make_utils import format_filename
    assert 'test' == str(format_filename('test', sca=1))
    assert 'test_wfi01' == str(format_filename('test_{}', sca=1))
    assert 'test_wfi01_foo' == str(format_filename(
        'test_{}_{bandpass}', sca=1, bandpass='foo'))
