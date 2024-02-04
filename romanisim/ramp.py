"""Ramp fitting routines.

The simulator need not actually fit any ramps, but we would like to do a good
job simulating the noise induced by ramp fitting.  That requires computing the
covariance matrix coming out of ramp fitting.  But that's actually a big part
of the work of ramp fitting.

There are a few different proposed ramp fitting algorithms, differing in their
weights.  The final derived covariances are all somewhat similarly difficult
to compute, however, since we ultimately end up needing to compute

.. math:: (A^T C^{-1} A)^{-1}

for the "optimal" case, or

.. math:: (A^T W^{-1} A)^{-1} A^T W^{-1} C W^{-1} A (A^T W^{-1} A)^{-1}

for some alternative weighting.

We start trying the "optimal" case below.

For the "optimal" case, a challenge is that we don't want to compute
:math:`C^{-1}` for every pixel individually.  Fortunately, we only
need :math:`(A^T C^{-1} A)^{-1}` (which is only a 2x2 matrix) for variances,
and only :math:`(A^T C^{-1} A)^{-1} A^T C^{-1}` for ramp fitting, which is 2xn.
Both of these matrices are effectively single parameter families, depending
after rescaling by the read noise only on the ratio of the read noise and flux.

So the routines in these packages construct these different matrices, store
them, and interpolate between them for different different fluxes and ratios.
"""

import numpy as np
from . import parameters, l1
import romanisim.ramp_fit_casertano
from scipy import interpolate
from astropy import units as u


def read_pattern_to_tbar(read_pattern):
    """Construct the mean times for each resultant from a read_pattern.

    Parameters
    ----------
    read_pattern : list[list]
        List of lists specifying the indices of the reads entering each
        resultant.

    Returns
    -------
    tbar : np.ndarray[n_resultant] (float)
        The mean time of the reads of each resultant.
    """
    read_time = parameters.read_time
    meantimes = read_time * np.array([np.mean(res) for res in read_pattern])
    # at some point I need to think hard about whether the first read has
    # slightly less exposure time than all other reads due to the read/reset
    # time being slightly less than the read time.
    return meantimes


def read_pattern_to_tau(read_pattern):
    """Construct the tau for each resultant from a read_pattern.

    .. math:: \\tau = \\overline{t} - (n - 1)(n + 1)\\delta t / 6n

    following Casertano (2022).

    Parameters
    ----------
    read_pattern : list[list]
        List of lists specifying the indices of the reads entering each
        resultant.

    Returns
    -------
    :math:`\\tau`
        A time scale appropriate for computing variances.
    """
    tij = l1.read_pattern_to_tij(read_pattern)
    nreads = np.array([len(x) for x in read_pattern])
    tau = np.array([np.sum((2 * (nn - np.arange(nn)) - 1) * tt) / nn ** 2
                    for (nn, tt) in zip(nreads, tij)])
    return tau


def construct_covar(read_noise, flux, read_pattern):
    """Constructs covariance matrix for first finite differences of unevenly
    sampled resultants.

    Parameters
    ----------
    read_noise : float
        The read noise (electrons)
    flux : float
        The electrons per second
    read_pattern : list[list]
        List of lists specifying the indices of the reads entering each
        resultant.

    Returns
    -------
    np.ndarray[n_resultant, n_resultant] (float)
        covariance matrix of first finite differences of unevenly sampled
        resultants.
    """
    tau = read_pattern_to_tau(read_pattern)
    tbar = read_pattern_to_tbar(read_pattern)
    nreads = np.array([len(x) for x in read_pattern])
    # from Casertano (2022), using Eqs 16, 19, and replacing with forward
    # differences.
    # diagonal -> (rn)^2/(1/N_i + 1/N_{i-1}) + f(tau_i + tau_{i-1} - 2t_{i-1}).
    # off diagonal: f(t_{i-1} - tau_{i-1}) - (rn)^2/N_{i-1}
    # further off diagonal: 0.
    diagonal = [[read_noise**2 / nreads[0] + flux * tau[0]],
                (read_noise**2 * (1 / nreads[1:] + 1 / nreads[:-1]) + flux * (
                    tau[1:] + tau[:-1] - 2 * tbar[:-1]))]
    cc = np.diag(np.concatenate(diagonal))

    off_diagonal = flux * (tbar[:-1] - tau[:-1]) - read_noise**2 / nreads[:-1]
    cc += np.diag(off_diagonal, 1)
    cc += np.diag(off_diagonal, -1)
    return cc.astype('f4')


def construct_ramp_fitting_matrices(covar, read_pattern):
    """Construct :math:`A^T C^{-1} A` and :math:`A^T C^{-1}`, the matrices
    needed to fit ramps from resultants.

    The matrices constructed are those needed for applying to differences
    of resultants; e.g., the results of resultants_to_differences.

    Parameters
    ----------
    covar : np.ndarray[n_resultant, n_resultant] (float)
        covariance of differences of resultants
    read_pattern : list[list]
        List of lists specifying the reads entering each resultant

    Returns
    -------
    atcinva, atcinv : np.ndarray[2, 2], np.ndarray[2, n_resultant] (float)
        :math:`A^T C^{-1} A` and :math:`A^T C^{-1}`, so that
        pedestal, flux = np.linalg.inv(atcinva).dot(atcinva.dot(differences))
    """

    aa = np.zeros((len(read_pattern), 2), dtype='f4')
    tbar = read_pattern_to_tbar(read_pattern)

    # pedestal; affects only 1st finite difference.
    aa[0, 0] = 1
    # slope; affects all finite differences
    aa[0, 1] = tbar[0]
    aa[1:, 1] = np.diff(tbar)
    cinv = np.linalg.inv(covar)
    # this won't be full rank if it's too small; we'll need some special
    # handling for nearly fully saturated cases, etc..
    atcinv = aa.T.dot(cinv)
    atcinva = atcinv.dot(aa)
    return atcinva, atcinv


def construct_ki_and_variances(atcinva, atcinv, covars):
    """Construct the :math:`k_i` weights and variances for ramp fitting.

    Following Casertano (2022), the ramp fit resultants are k.dot(differences),
    where :math:`k=(A^T C^{-1} A)^{-1} A^T C^{-1}`, and differences is the
    result of resultants_to_differences(resultants).  Meanwhile the variances
    are :math:`k C k^T`.  This function computes these k and variances.

    Parameters
    ----------
    atcinva : np.ndarray[2, 2] (float)
        :math:`A^T C^{-1} A` from construct_ramp_fitting_matrices
    atcinv : np.ndarray[2, n_resultant] (float)
        :math:`A^T C^{-1}` from construct_ramp_fitting_matrices
    covars : list[np.ndarray[n_resultant, n_resultant]]
        covariance matrices to contract against :math:`k` to compute variances

    Returns
    -------
    k : np.ndarray[2, n_resultant]
        :math:`k = (A^T C^{-1} A)^-1 A^T C^{-1}` from Casertano (2022)
    variances : list[np.ndarray[2, 2]] (float)
        :math:`k C_i k^T` for different covariance matrices C_i
        supplied in covars
    """

    k = np.linalg.inv(atcinva).dot(atcinv)
    variances = [k.dot(c).dot(k.T) for c in covars]
    return k, variances


def ki_and_variance_grid(read_pattern, flux_on_readvar_pts):
    """Construct a grid of :math:`k` and covariances for the values of
    flux_on_readvar.

    The :math:`k` and corresponding covariances needed to do ramp fitting
    form essentially a one dimensional family in the flux in the ramp divided
    by the square of the read noise.  This function constructs these quantities
    for a large number of different flux / read_noise^2 to be used in
    interpolation.

    Parameters
    ----------
    read_pattern : list[list] (int)
        a list of lists of the indices of the reads entering each resultant
    flux_on_readvar_pts : array_like (float)
        values of flux / read_noise**2 for which :math:`k` and variances are
        desired.

    Returns
    -------
    kigrid : np.ndarray[len(flux_on_readvar_pts), 2, n_resultants] (float)
        :math:`k` for each value of flux_on_readvar_pts
    vargrid : np.ndarray[len(flux_on_readvar_pts), n_covar, 2, 2] (float)
        covariance of pedestal and slope corresponding to each value of
        flux_on_readvar_pts.  n_covar = 3, for the contributions from
        read_noise, Poisson noise, and the sum.
    """
    # the ramp fitting covariance matrices make a one-dimensional
    # family.  If we divide out the read variance, the single parameter
    # is flux / read_noise**2
    cc_rn = construct_covar(1, 0, read_pattern)
    cc_flux = construct_covar(0, 1, read_pattern)
    outki = []
    outvar = []
    for flux_on_readvar in flux_on_readvar_pts:
        cc_flux_scaled = cc_flux * flux_on_readvar
        atcinva, atcinv = construct_ramp_fitting_matrices(
            cc_rn + cc_flux_scaled, read_pattern)
        covars = [cc_rn, cc_flux_scaled, cc_rn + cc_flux_scaled]
        ki, variances = construct_ki_and_variances(atcinva, atcinv, covars)
        outki.append(ki)
        outvar.append(variances)
    return np.array(outki), np.array(outvar)


# okay, how do we get what we want?  we want to input a bunch of
# read_noise and fluxes, and get out a bunch of ki and outvar.
# it feels like we can just set up a bunch of 1D interpolators?

class RampFitInterpolator:
    """Ramp fitting tool aiding efficient fitting of large number of ramps.

    The basic idea is that for a given image, ignoring cosmic rays or saturated
    pixels, the ramp fitting parameters are just a linear combination of the
    resultants.  The weights of this linear combination are a single parameter
    family in the flux in the ramp divided by the read variance.  So rather than
    explicitly calculating those weights for each pixel, we can up front calculate
    them overa  grid in the flux over the read variance, and interpolate off that
    grid for each point.  That can all be done in a vectorized way, allowing one
    to avoid doing something like a matrix inverse for each of a 16 million pixels.

    The tool pre-calculates the grid and interpolators it needs at initialization,
    and then uses the results of that calculation when invoked to get the weights
    :math:`k` or variances.  The expectation is that most users just initialize
    and then call the fit_ramps method.
    """
    def __init__(self, read_pattern, flux_on_readvar_pts=None):
        """Construct a RampFitInterpolator for a read_pattern and a grid of
        flux/read_noise**2.

        Parameters
        ----------
        read_pattern : list[list] (int)
            list of lists of indices of reads entering each resultant
        flux_on_readvar_pts : np.ndarray (float)
            flux / read_noise**2 points at which to compute ramp fitting
            matrices.
            if None, a default grid will be used that should cover all reasonable
            values, from the read variance being 100k larger to 100k smaller than
            the electrons per second.
        """
        if flux_on_readvar_pts is None:
            self.flux_on_readvar_pts = 10.**(
                np.linspace(-5, 5, 200, dtype='f4'))
        else:
            self.flux_on_readvar_pts = flux_on_readvar_pts
        ki, var = ki_and_variance_grid(read_pattern, self.flux_on_readvar_pts)
        self.read_pattern = read_pattern
        self.ki_vals = ki
        self.var_vals = var
        self.ki_interpolator = interpolate.CubicSpline(
            self.flux_on_readvar_pts, self.ki_vals)
        self.var_interpolator = interpolate.CubicSpline(
            self.flux_on_readvar_pts, self.var_vals)

        # interp1d has a lot of options, but the timing difference on my system
        # is swamped by the cpu boosting up or down?
        # most of the time in this function goes to evaluating the interpolators,
        # so it's worth playing around a little to see if there's a win there.
        # self.ki_interpolator = interpolate.interp1d(
        #     self.flux_on_readvar_pts, self.ki_vals, axis=0)
        # self.var_interpolator = interpolate.interp1d(
        #     self.flux_on_readvar_pts, self.var_vals, axis=0)

    def ki(self, flux, read_noise):
        """Compute :math:`k`, the weights for the linear combination of
        resultant differences for optimal measurement of ramp pedestal and
        slope.

        Parameters
        ----------
        flux : array_like (float)
            Estimate of electrons per second in ramp
        read_noise : array_like (float)
            read_noise in ramp.  Must be broadcastable with flux.

        Returns
        -------
        ki : array_like[..., 2, n_resultant] (float)
            :math:`k`, weights of differences in linear combination of
            ramp pixels
        """
        # clip points outside range to edges.
        fluxonreadvar = flux / read_noise**2
        if isinstance(fluxonreadvar, u.Quantity):
            unit = fluxonreadvar.unit
        else:
            unit = 1
        fluxonreadvar = np.clip(
            fluxonreadvar, self.flux_on_readvar_pts[0] * unit,
            self.flux_on_readvar_pts[-1] * unit)

        return self.ki_interpolator(fluxonreadvar).astype('f4')

    def variances(self, flux, read_noise):
        """Compute the variances of ramp fit parameters.

        Parameters
        ----------
        flux : array_like (float)
            Estimate of electrons per second in ramp
        read_noise : array_like (float)
            read_noise in ramp.  Must be broadcastable with flux.

        Returns
        -------
        variances : array_like[..., 3, 2, 2] (float)
            covariance of ramp fit parameters, for read noise, poisson noise,
            and the total noise
        """
        # clip points outside range to edges.
        fluxonreadvar = flux / read_noise**2
        if isinstance(fluxonreadvar, u.Quantity):
            unit = fluxonreadvar.unit
        else:
            unit = 1
        fluxonreadvar = np.clip(
            fluxonreadvar, self.flux_on_readvar_pts[0] * unit,
            self.flux_on_readvar_pts[-1] * unit)
        var = self.var_interpolator(fluxonreadvar).astype('f4')
        read_noise = np.array(read_noise)
        read_noise = read_noise.reshape(
            read_noise.shape + (1,) * (len(var.shape) - len(read_noise.shape)))
        var *= read_noise**2
        return var

    def fit_ramps(self, resultants, read_noise, fluxest=None):
        """Fit ramps for a set of resultants and their read noise.

        Does not handle partial ramps (i.e., broken due to CRs).

        Parameters
        ----------
        resultants : np.ndarray[n_resultants, nx, ny] (numeric)
            Resultants to fit
        read_noise : float or array_like like resultants
            read noise in array
        fluxest : float or array_like like resultants
            Initial estimate of flux in each ramp, in electrons per second.
            If None, estimated from the median flux differences between
            resultants.

        Returns
        -------
        par : np.ndarray[nx, ny, 2] (float)
            the best fit pedestal and slope for each pixel

        var : np.ndarray[nx, ny, 3, 2, 2] (float)
            the covariance matrix of par, for each of three noise terms:
            the read noise, Poisson source noise, and total noise.
        """
        tbar = read_pattern_to_tbar(self.read_pattern)
        dtbar = np.diff(tbar).astype('f4')
        differences = resultants_to_differences(resultants)
        if fluxest is None:
            dtbar_reshape = dtbar.reshape(
                (dtbar.shape[0],) + (1,) * len(differences.shape[1:]))
            fluxest = np.median(differences[1:] / dtbar_reshape, axis=0)
        if (np.ndim(fluxest) == 0) and (np.ndim(resultants) > 1):
            fluxest = fluxest * np.ones(resultants.shape[1:])
        ki = self.ki(fluxest, read_noise)
        par = np.einsum('...cd,d...->...c', ki, differences)
        var = self.variances(fluxest, read_noise)
        return par, var


def resultants_to_differences(resultants):
    """Convert resultants to their finite differences.

    This is essentially np.diff(...), but retains the first
    resultant.  The resulting structure has tri-diagonal covariance,
    which can be a little useful.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, nx, ny] (float)
        The resultants

    Returns
    -------
    differences : np.ndarray[n_resultant, nx, ny] (float)
        Differences of resultants
    """
    return np.vstack([resultants[0][None, :],
                      np.diff(resultants, axis=0)])


# how do I demonstrate things are right?
# let's just make a bunch of simulations of the same ramp, compute
# an empirical covariance matrix, and compare it with an analytic one?
def simulate_many_ramps(ntrial=100, flux=100, readnoise=5, read_pattern=None):
    """Simulate many ramps with a particular flux, read noise, and read_pattern.

    To test ramp fitting, it's useful to be able to simulate a large number
    of ramps that are identical up to noise.  This function does that.

    Parameters
    ----------
    ntrial : int
        number of ramps to simulate
    flux : float
        flux in electrons / s
    read_noise : float
        read noise in electrons
    read_pattern : list[list] (int)
        list of lists giving indices of reads entering each resultant

    Returns
    -------
    read_pattern : list[list] (int)
        read_pattern used
    flux : float
        flux used
    readnoise : float
        read noise used
    resultants : np.ndarray[n_resultant, ntrial] (float)
        simulated resultants
    """
    from . import l1

    if read_pattern is None:
        read_pattern = [
            [1, 2, 3, 4], [5], [6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [19, 20, 21],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]]
    nread = np.array([len(x) for x in read_pattern])
    tij = l1.read_pattern_to_tij(read_pattern)
    totcounts = np.random.poisson(flux * np.max(np.concatenate(tij)), ntrial)
    resultants, dq = l1.apportion_counts_to_resultants(totcounts, tij)
    resultants += np.random.randn(len(read_pattern), ntrial) * (
        readnoise / np.sqrt(nread)).reshape(len(read_pattern), 1)
    return (read_pattern, flux, readnoise, resultants)


def fit_ramps_casertano(resultants, dq, read_noise, read_pattern):
    """Fit ramps following Casertano+2022, including averaging partial ramps.

    Ramps are broken where dq != 0, and fits are performed on each sub-ramp.
    Resultants containing multiple ramps have their ramp fits averaged using
    inverse variance weights based on the variance in the individual slope fits
    due to read noise.

    Parameters
    ----------
    resultants : np.ndarry[nresultants, ...]
        the resultants in electrons
    dq : np.ndarry[nresultants, ...]
        the dq array.  dq != 0 implies bad pixel / CR.
    read noise: float
        the read noise in electrons
    read_pattern : list[list[int]]
        list of lists giving indices of reads entering each resultant

    Returns
    -------
    par : np.ndarray[..., 2] (float)
        the best fit pedestal and slope for each pixel
    var : np.ndarray[..., 3, 2, 2] (float)
        the covariance matrix of par, for each of three noise terms:
        the read noise, Poisson source noise, and total noise.
    """

    resultants_unit = getattr(resultants, 'unit', None)
    if resultants_unit is not None:
        resultants = resultants.to(u.electron).value

    resultants = np.array(resultants).astype('f4')

    dq = np.array(dq).astype('i4')

    if np.ndim(read_noise) <= 1:
        read_noise = read_noise * np.ones(resultants.shape[1:])
    read_noise = np.array(read_noise).astype('f4')

    origshape = resultants.shape
    if len(resultants.shape) == 1:
        # single ramp.
        resultants = resultants.reshape(origshape + (1,))
        dq = dq.reshape(origshape + (1,))
        read_noise = read_noise.reshape(origshape[1:] + (1,))

    rampfitdict = romanisim.ramp_fit_casertano.fit_ramps(
        resultants.reshape(resultants.shape[0], -1),
        dq.reshape(resultants.shape[0], -1),
        read_noise.reshape(-1),
        read_pattern)

    par = np.zeros(resultants.shape[1:] + (2,), dtype='f4')
    var = np.zeros(resultants.shape[1:] + (3, 2, 2), dtype='f4')

    npix = resultants.reshape(resultants.shape[0], -1).shape[1]
    # we need to do some averaging to merge the results in each ramp.
    # inverse variance weights based on slopereadvar
    weight = ((rampfitdict['slopepoissonvar'] != 0) / (
        rampfitdict['slopereadvar'] + (rampfitdict['slopereadvar'] == 0)))
    weight = np.clip(weight, 0, 10**10)  # gracefully? handle read noise = 0 case.
    totweight = np.bincount(rampfitdict['pix'], weights=weight, minlength=npix)
    totval = np.bincount(rampfitdict['pix'],
                         weights=weight * rampfitdict['slope'],
                         minlength=npix)
    # fill in the averaged slopes
    par.reshape(npix, 2)[:, 1] = (
        totval / (totweight + (totweight == 0)))

    # read noise variances
    totval = np.bincount(
        rampfitdict['pix'], weights=weight ** 2 * rampfitdict['slopereadvar'],
        minlength=npix)
    var.reshape(npix, 3, 2, 2)[:, 0, 1, 1] = (
        totval / (totweight ** 2 + (totweight == 0)))
    # poisson noise variances
    totval = np.bincount(
        rampfitdict['pix'],
        weights=weight ** 2 * rampfitdict['slopepoissonvar'], minlength=npix)
    var.reshape(npix, 3, 2, 2)[..., 1, 1, 1] = (
        totval / (totweight ** 2 + (totweight == 0)))

    # multiply Poisson term by flux.  Clip at zero; no negative Poisson variances.
    var[..., 1, 1, 1] *= np.clip(par[..., 1], 0, np.inf)
    var[..., 2, 1, 1] = var[..., 0, 1, 1] + var[..., 1, 1, 1]

    if resultants.shape != origshape:
        par = par[0]
        var = var[0]

    if resultants_unit is not None:
        par = par * resultants_unit

    return par, var


def fit_ramps_casertano_no_dq(resultants, read_noise, read_pattern):
    """Fit ramps following Casertano+2022, only using full ramps.

    This is a simpler implementation of fit_ramps_casertano, which doesn't
    address the case of partial ramps broken by CRs.  This case is easier
    and can be done reasonably efficiently in pure python; results can be
    compared with fit_ramps_casertano in for the case of unbroken ramps.

    Parameters
    ----------
    resultants : np.ndarry[nresultants, npixel]
        the resultants in electrons
    read noise: float
        the read noise in electrons
    read_pattern : list[list[int]]
        list of lists giving indices of reads entering each resultant

    Returns
    -------
    par : np.ndarray[nx, ny, 2] (float)
        the best fit pedestal and slope for each pixel
    var : np.ndarray[nx, ny, 3, 2, 2] (float)
        the covariance matrix of par, for each of three noise terms:
        the read noise, Poisson source noise, and total noise.
    """
    nadd = len(resultants.shape) - 1
    if np.ndim(read_noise) <= 1:
        read_noise = np.array(read_noise).reshape((1,) * nadd)
    smax = resultants[-1]
    s = smax / np.sqrt(read_noise**2 + smax)  # Casertano+2022 Eq. 44
    ptable = np.array([  # Casertano+2022, Table 2
        [-np.inf, 0], [5, 0.4], [10, 1], [20, 3], [50, 6], [100, 10]])
    pp = ptable[np.searchsorted(ptable[:, 0], s) - 1, 1]
    nn = np.array([len(x) for x in read_pattern])  # number of reads in each resultant
    tbar = read_pattern_to_tbar(read_pattern)
    tau = read_pattern_to_tau(read_pattern)
    tbarmid = (tbar[0] + tbar[-1]) / 2
    if nadd > 0:
        newshape = ((-1,) + (1,) * nadd)
        nn = nn.reshape(*newshape)
        tbar = tbar.reshape(*newshape)
        tau = tau.reshape(*newshape)
        tbarmid = tbarmid.reshape(*newshape)
    ww = (  # Casertano+22, Eq. 45
        (1 + pp)[None, ...] * nn
        / (1 + pp[None, ...] * nn)
        * np.abs(tbar - tbarmid) ** pp[None, ...])

    # Casertano+22 Eq. 35
    f0 = np.sum(ww, axis=0)
    f1 = np.sum(ww * tbar, axis=0)
    f2 = np.sum(ww * tbar**2, axis=0)
    # Casertano+22 Eq. 36
    dd = f2 * f0 - f1 ** 2
    bad = dd == 0
    dd[bad] = 1
    # Casertano+22 Eq. 37
    kk = (f0[None, ...] * tbar - f1[None, ...]) * ww / (
        dd[None, ...])
    # shape: [n_resultant, ny, nx]
    ff = np.sum(kk * resultants, axis=0)  # Casertano+22 Eq. 38
    # Casertano+22 Eq. 39
    vr = np.sum(kk**2 / nn, axis=0) * read_noise**2
    # Casertano+22 Eq. 40
    vs1 = np.sum(kk**2 * tau, axis=0)
    vs2inner = np.cumsum(kk * tbar, axis=0)
    vs2inner = np.concatenate([0 * vs2inner[0][None, ...], vs2inner[:-1, ...]], axis=0)
    vs2 = 2 * np.sum(vs2inner * kk, axis=0)
    # sum_{i=1}^{j-1} K_i \bar{t}_i
    # this is the inner of the two sums in the 2nd term of Eq. 40
    # Casertano+22 has some discussion of whether it's more efficient to do
    # this as an explicit double sum or to construct the inner sum separately.
    # We've made a lot of other quantities that are [nr, ny, nx] in size,
    # so I don't feel bad about making another.  Clearly a memory optimized
    # code would work a lot harder to reuse a lot of variables above!

    vs = (vs1 + vs2) * ff
    vs = np.clip(vs, 0, np.inf)
    # we can estimate negative flux, but we really shouldn't add variance for
    # that case!

    # match return values from RampFitInterpolator.fit_ramps
    # we haven't explicitly calculated here the pedestal, its
    # uncertainty, or covariance terms.  We just fill
    # with zeros.

    par = np.zeros(ff.shape + (2,), dtype='f4')
    var = np.zeros(ff.shape + (3, 2, 2), dtype='f4')
    par[..., 1] = ff
    var[..., 0, 1, 1] = vr
    var[..., 1, 1, 1] = vs
    var[..., 2, 1, 1] = vr + vs

    return par, var
