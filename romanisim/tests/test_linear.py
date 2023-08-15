"""Unit tests for linear module.

Routines tested:
- validate_times
- tij_to_pij
- apportion_counts_to_resultants
- add_read_noise_to_resultants
- make_asdf
- ma_table_to_tij
- make_l1
"""
import os
import pytest
import numpy as np
import galsim
import copy
from astropy import units as u
import crds

import roman_datamodels
import roman_datamodels.maker_utils as maker_utils
from romanisim import nonlinearity, parameters, wcs, catalog, image, util



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

INVERSE = [True, False]


def test_linear_apply():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.asfarray([1.0, 0.7, 3.0e-6, 5.0e-12])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    gain = 4.0 * u.electron / u.DN
    counts[0, 0] = counts[0, 99] = counts[99, 0] = counts[99, 99] = 11.0

    linearity = nonlinearity.NL(lin_coeffs, gain)

    res2 = linearity.apply(counts, electrons=True)

    assert res2[0,0] == res2[0,99]
    assert res2[99, 0] == res2[99, 99]
    assert res2[0, 0] == 2 * res2[99, 99]


@pytest.mark.parametrize("inverse", INVERSE)
def test_repair_coeffs(inverse):
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.asfarray([1.0, 0.7, 3.0e-6, 5.0e-12])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    lin_coeffs[:, 1, 1] *= 0
    lin_coeffs[2, 22, 22] = np.nan
    gain = 4.0 * u.electron / u.DN

    linearity = nonlinearity.NL(lin_coeffs, gain, inverse)

    res = linearity.apply(counts)

    assert res[1, 1] == counts[1, 1]
    assert res[22, 22] == counts[22, 22]
    assert np.all(res[23:,23:] != counts[23:,23:])

def test_electrons():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.asfarray([1.0, 0.7, 3.0e-6, 5.0e-12])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))
    lin_coeffs[:, 0:50, :] *= 2.0
    gain = 4.0 * u.electron / u.DN

    linearity = nonlinearity.NL(lin_coeffs, gain)

    res = linearity.apply(counts)

    res_elec = linearity.apply(gain * counts, electrons=True)

    assert np.all(res_elec[:] == gain * res[:])
    assert res_elec.unit == u.electron / u.DN
    assert not hasattr(res, "unit")

def test_reverse():
    counts = np.random.poisson(100, size=(100, 100))
    coeffs = np.asfarray([1.0, 0.7, 3.0e-6, 5.0e-12])
    lin_coeffs = np.tile(coeffs[:, np.newaxis, np.newaxis], (1, 100, 100))

    lin_coeffs[:, 0:50, :] *= 2.0
    rev_lin_coeffs = lin_coeffs[::-1, ...]
    gain = 4.0 * u.electron / u.DN

    linearity = nonlinearity.NL(lin_coeffs, gain)
    rev_linearity = nonlinearity.NL(rev_lin_coeffs, gain)

    res = linearity.apply(counts)
    res_rev = rev_linearity.apply(counts, reversed=True)

    assert np.all(res_rev[:] == res[:])

@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason=(
        "Roman CRDS servers are not currently available outside the internal network"
    ),
)
def test_inverse_then_linearity():
    # Test that applying inverse linearity and then linearity returns the results to
    # the original value

    # Set up parameters for simulation run
    galsim.roman.n_pix = 4088
    metadata = copy.deepcopy(parameters.default_parameters_dictionary)
    metadata['instrument']['detector'] = 'WFI07'
    metadata['instrument']['optical_element'] = 'F158'
    metadata['exposure']['ma_table_number'] = 1

    twcs = wcs.get_wcs(metadata, usecrds=True)
    rd_sca = twcs.toWorld(galsim.PositionD(
        galsim.roman.n_pix / 2, galsim.roman.n_pix / 2))

    cat = catalog.make_dummy_table_catalog(
        rd_sca, bandpasses=[metadata['instrument']['optical_element']], nobj=1000)

    rng = galsim.UniformDeviate(None)

    meta = maker_utils.mk_common_meta()
    meta["photometry"] = maker_utils.mk_photometry()

    for key in parameters.default_parameters_dictionary.keys():
        meta[key].update(parameters.default_parameters_dictionary[key])

    util.add_more_metadata(meta)

    for key in metadata.keys():
        meta[key].update(metadata[key])

    image_node = maker_utils.mk_level2_image()
    image_node['meta'] = meta
    image_mod = roman_datamodels.datamodels.ImageModel(image_node)

    refnames_lst = ['inverselinearity', 'linearity']

    reffiles = crds.getreferences(
        image_mod.get_crds_parameters(), reftypes=refnames_lst,
        observatory='roman')

    inverse_linearity_model = roman_datamodels.datamodels.InverselinearityRefModel(
        reffiles['inverselinearity'])
    linearity_model = roman_datamodels.datamodels.LinearityRefModel(
        reffiles['linearity'])

    gain = 1.0

    inverse_linearity = nonlinearity.NL(
        inverse_linearity_model.coeffs[:,0:4088,0:4088], gain=gain)
    linearity = nonlinearity.NL(
        linearity_model.coeffs[:,0:4088,0:4088], gain=gain)



    level_0, simcatobj = image.simulate(
        metadata, cat, usecrds=True,
        webbpsf=True, level=0,
        rng=rng)

    level_0_inv = inverse_linearity.apply(level_0['data'])

    level_0_lin = linearity.apply(level_0_inv)

    # Divide initial counts before and after inverse & NL application
    div = level_0['data'][np.isfinite(level_0['data'])] / level_0_lin[np.isfinite(level_0_lin)]

    assert np.isclose(np.average(div), 1.0, atol=0.1)

    pass
