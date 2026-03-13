
"""
Unit tests for ramp-fitting functions.  Tested routines:
* read_pattern_to_tbar
* read_pattern_to_tau
* construct_covar
* construct_ramp_fitting_matrices
* construct_ki_and_variances
* ki_and_variance_grid
* RampFitInterpolator
  * __init__
  * ki
  * variances
  * fit_ramps
* resultants_to_differences
* simulate_many_ramps
"""

import numpy as np
from romanisim import ramp
from romanisim.models import parameters


def test_ramp(test_table=None):
    if test_table is None:
        test_table = [[1 + x for x in range(3)],
                      [4 + x for x in range(5)],
                      [10], [11], [21 + x for x in range(3)]]
    tbar = ramp.read_pattern_to_tbar(test_table)
    read_time = parameters.read_time
    assert np.allclose(
        tbar, [read_time * np.mean(x) for x in test_table])
    tau = ramp.read_pattern_to_tau(test_table)
    # this is kind of just a defined function; I don't have a really good
    # test for this without just duplicating code.
    nreads = np.array([len(x) for x in test_table])
    assert np.allclose(
        tau, tbar - (nreads - 1) * (nreads + 1) * read_time / 6 / nreads)
    read_noise = 3
    flux = 100
    covar1 = ramp.construct_covar(read_noise, 0, test_table)
    covar2 = ramp.construct_covar(0, flux, test_table)
    covar3 = ramp.construct_covar(read_noise, flux, test_table)
    assert np.allclose(covar1 + covar2, covar3)
    for c in (covar1, covar2, covar3):
        assert np.allclose(c, c.T)
    read_diag = np.concatenate([
        [read_noise**2 / nreads[0]],
        read_noise**2 * (1 / nreads[:-1] + 1 / nreads[1:])])
    read_offdiag = -read_noise**2 / nreads[:-1]
    flux_diag = np.concatenate([
        [flux * tau[0]], flux * (tau[:-1] + tau[1:] - 2 * tbar[:-1])])
    flux_offdiag = flux * (tbar[:-1] - tau[:-1])
    assert np.allclose(np.diag(covar1), read_diag)
    assert np.allclose(np.diag(covar1, 1), read_offdiag)
    assert np.allclose(np.diag(covar2), flux_diag)
    assert np.allclose(np.diag(covar2, 1), flux_offdiag)
    atcinva, atcinv = ramp.construct_ramp_fitting_matrices(covar3, test_table)
    aa = np.zeros((len(test_table), 2), dtype='f4')
    aa[0, 0] = 1
    aa[0, 1] = tbar[0]
    aa[1:, 1] = np.diff(tbar)
    assert np.allclose(aa.T.dot(np.linalg.inv(covar3)).dot(aa), atcinva)
    assert np.allclose(aa.T.dot(np.linalg.inv(covar3)), atcinv)
    param = np.array([10, 100], dtype='f4')
    yy = aa.dot(param)
    param2 = np.linalg.inv(atcinva).dot(atcinv.dot(yy))
    assert np.allclose(param, param2, rtol=1e-3)
    ki, var = ramp.construct_ki_and_variances(
        atcinva, atcinv, [covar1, covar2, covar3])
    assert np.allclose(var[2], np.linalg.inv(atcinva), rtol=1e-3)
    assert np.allclose(ki, np.linalg.inv(atcinva).dot(atcinv), rtol=1e-3)
    npts = 101
    flux_on_readvar_pts = 10.**(np.linspace(-5, 5, npts))
    kigrid, vargrid = ramp.ki_and_variance_grid(
        test_table, flux_on_readvar_pts)
    assert np.all(
        np.array(kigrid.shape) == np.array([npts, 2, len(test_table)]))
    assert np.all(
        np.array(vargrid.shape) == np.array([npts, 3, 2, 2]))
    rcovar = ramp.construct_covar(1, 0, test_table)
    fcovar = ramp.construct_covar(0, 1, test_table)
    for i in np.arange(0, npts, npts // 3):
        acovar = ramp.construct_covar(1, flux_on_readvar_pts[i], test_table)
        atcinva, atcinv = ramp.construct_ramp_fitting_matrices(
            acovar, test_table)
        scale = flux_on_readvar_pts[i]
        ki, var = ramp.construct_ki_and_variances(
            atcinva, atcinv, [rcovar, fcovar * scale, acovar])
        assert np.allclose(kigrid[i], ki, atol=1e-5, rtol=1e-4)
        assert np.allclose(vargrid[i], var, atol=1e-5, rtol=1e-4)
    fitter = ramp.RampFitInterpolator(test_table, flux_on_readvar_pts)
    ki = fitter.ki(flux, read_noise)
    var = fitter.variances(flux, read_noise)
    fluxes = np.array([10, 100, 1000, 1, 2, 3, 0])
    pedestals = np.array([-10, 0, 10, -1, 0, 1, 2])
    resultants = (fluxes.reshape(1, -1) * tbar.reshape(-1, 1)
                  + pedestals.reshape(1, -1))
    from functools import partial
    rampfitters = [
        fitter.fit_ramps,
        partial(ramp.fit_ramps_casertano_no_dq, read_pattern=test_table),
        partial(ramp.fit_ramps_casertano, read_pattern=test_table,
                dq=resultants * 0)
    ]
    for fitfun in rampfitters:
        par, var = fitfun(resultants=resultants, read_noise=read_noise)
        assert np.allclose(par[:, 1], fluxes, atol=1e-6)
        if not np.all(par[:, 0]) == 0:
            assert np.allclose(par[:, 0], pedestals, atol=2e-2)

    # compare single ramp and multi-ramp versions
    p1, v1 = ramp.fit_ramps_casertano(resultants, resultants * 0, read_noise,
                                      test_table)
    p2, v2 = ramp.fit_ramps_casertano(resultants[:, 0], resultants[:, 0] * 0,
                                      read_noise, test_table)
    assert np.all(np.isclose(p1[0], p2))
    assert np.all(np.isclose(v1[0], v2))


def test_hard_ramps():
    read_patterns = list()
    read_patterns.append([[1], [2]])  # simple ramp
    read_patterns.append([list(range(1, 101)), list(range(101, 201))])
    read_patterns.append([[x] for x in np.arange(1, 101)])  # big ramp
    read_patterns.append([[1], [101]])  # big skip
    for tab in read_patterns:
        test_ramp(tab)


def test_simulated_ramps():
    ntrial = 100000
    read_pattern, flux, read_noise, resultants = ramp.simulate_many_ramps(
        ntrial=ntrial)
    fitter = ramp.RampFitInterpolator(read_pattern)
    par, var = fitter.fit_ramps(resultants, read_noise)
    par2, var2 = fitter.fit_ramps(resultants, read_noise, fluxest=flux)
    # in the simulation, the true flux was flux, so we expect this
    # to be ~Gaussian distributed about 0 with variance var.
    # the offset was 0.
    for p, v in [[par, var], [par2, var2]]:
        chi2dof_slope = np.sum((p[:, 1] - flux)**2 / v[:, 2, 1, 1]) / ntrial
        chi2dof_pedestal = np.sum((p[:, 0] - 0)**2 / v[:, 2, 0, 0]) / ntrial
        assert np.abs(chi2dof_slope - 1) < 0.03
        assert np.abs(chi2dof_pedestal - 1) < 0.03
    # It's not clear what level of precision to demand here.  This should
    # not actually give a value consistent with a normal chi^2 distribution
    # because the Poisson noise is Poisson distributed, but we are treating
    # it as Gaussian distributed in the ramp fitting.  So we kind of just want
    # good rather than perfect.  The above requires that the sigmas be right
    # at the 5% level, which isn't terrible.  Presumably the accuracy
    # of the simulation is best for very high and very low count rates,
    # as in those two limits either the Poisson counts can be very closely
    # approximated as Gaussians, or the Poisson counts are not important.
    # Still, the default settings do give something very close to chi^2/dof
    # = 1.  Running 10**6 samples, we get chi^2/dof ~ 1 +/- 0.001 or so.
    # This test depends on random numbers and may eventually fail, but
    # the 0.03 margin is enough that it should very (_very_) rarely fail
    # absent code changes.

    par, var = ramp.fit_ramps_casertano(
        resultants, resultants * 0, read_noise, read_pattern)
    chi2dof_slope = np.sum((par[:, 1] - flux)**2 / var[:, 2, 1, 1]) / ntrial
    assert np.abs(chi2dof_slope - 1) < 0.03

    # now let's mark a bunch of the ramps as compromised.
    bad = np.random.uniform(size=resultants.shape) > 0.7
    dq = resultants * 0 + bad
    par, var = ramp.fit_ramps_casertano(
        resultants, dq, read_noise, read_pattern)
    # only use okay ramps
    # ramps passing the below criterion have at least two adjacent valid reads
    # i.e., we can make a measurement from them.
    m = np.sum((dq[1:, :] == 0) & (dq[:-1, :] == 0), axis=0) != 0
    chi2dof_slope = np.sum((par[m, 1] - flux)**2 / var[m, 2, 1, 1]) / np.sum(m)
    assert np.abs(chi2dof_slope - 1) < 0.03
    assert np.all(par[~m, 1] == 0)
    assert np.all(var[~m, 1] == 0)


def test_resultants_to_differences():
    resultants = np.array([[10, 11, 12, 13, 14, 15]], dtype='f4').T
    differences = ramp.resultants_to_differences(resultants)
    assert np.allclose(differences, np.array([[10, 1, 1, 1, 1, 1]]).T)
