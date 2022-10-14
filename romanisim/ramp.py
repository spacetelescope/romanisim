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
from . import parameters
from scipy import interpolate


def ma_table_to_tbar(ma_table):
    """Construct the mean times for each resultant from an ma_table.

    Parameters
    ----------
    ma_table : list[list]
        List of lists specifying the first read and the number of reads in each
        resultant.

    Returns
    -------
    tbar : np.ndarray[n_resultant] (float)
        The mean time of the reads of each resultant.
    """
    firstreads = np.array([x[0] for x in ma_table])
    nreads = np.array([x[1] for x in ma_table])
    read_time = parameters.read_time
    meantimes = read_time * firstreads + read_time * (nreads - 1)/2
    # at some point I need to think hard about whether the first read has
    # slightly less exposure time than all other reads due to the read/reset
    # time being slightly less than the read time.
    return meantimes


def ma_table_to_tau(ma_table):
    """Construct the tau for each resultant from an ma_table.

    .. math:: \\tau = \overline{t} - (n - 1)(n + 1)\delta t / 6n

    following Casertano (2022).

    Parameters
    ----------
    ma_table : list[list]
        List of lists specifying the first read and the number of reads in each
        resultant.

    Returns
    -------
    :math:`\\tau`
        A time scale appropriate for computing variances.
    """

    meantimes = ma_table_to_tbar(ma_table)
    nreads = np.array([x[1] for x in ma_table])
    read_time = parameters.read_time
    return meantimes - (nreads - 1)*(nreads + 1)*read_time/6/nreads


def construct_covar(read_noise, flux, ma_table):
    """Constructs covariance matrix for first finite differences of unevenly
    sampled resultants.

    Parameters
    ----------
    read_noise : float
        The read noise
    flux : float
        The electrons per second (same units as read_noise)
    ma_table : list[list]
        List of lists specifying the first read and the number of reads in each
        resultant.

    Returns
    -------
    np.ndarray[n_resultant, n_resultant] (float)
        covariance matrix of first finite differences of unevenly sampled
        resultants.
    """
    read_time = parameters.read_time
    tau = ma_table_to_tau(ma_table)
    tbar = ma_table_to_tbar(ma_table)
    nreads = np.array([x[1] for x in ma_table])
    # from Casertano (2022), using Eqs 16, 19, and replacing with forward
    # differences.
    # diagonal -> (rn)^2/(1/N_i + 1/N_{i-1}) + f(tau_i + tau_{i-1} - 2t_{i-1}).
    # off diagonal: f(t_{i-1} - tau_{i-1}) - (rn)^2/N_{i-1}
    # further off diagonal: 0.
    diagonal = [[read_noise**2 / nreads[0] + flux*tau[0]],
                (read_noise**2 * (1/nreads[1:] + 1/nreads[:-1])  +
                 flux*(tau[1:] + tau[:-1] - 2*tbar[:-1]))]
    cc = np.diag(np.concatenate(diagonal))

    off_diagonal = flux*(tbar[:-1] - tau[:-1]) - read_noise**2/nreads[:-1]
    cc += np.diag(off_diagonal, 1)
    cc += np.diag(off_diagonal, -1)
    return cc.astype('f4')


def construct_ramp_fitting_matrices(covar, ma_table):
    """Construct :math:`A^T C^{-1} A` and :math:`A^T C^{-1}`, the matrices
    needed to fit ramps from resultants.

    The matrices constructed are those needed for applying to differences
    of resultants; e.g., the results of resultants_to_differences.

    Parameters
    ----------
    covar : np.ndarray[n_resultant, n_resultant] (float)
        covariance of differences of resultants
    ma_table : list[list] giving first read number and number of reads in each
        resultant

    Returns
    -------
    atcinva, atcinv : np.ndarray[2, 2], np.ndarray[2, n_resultant] (float)
        :math:`A^T C^{-1} A` and :math:`A^T C^{-1}`, so that
        pedestal, flux = np.linalg.inv(atcinva).dot(atcinva.dot(differences))
    """

    aa = np.zeros((len(ma_table), 2), dtype='f4')
    tbar = ma_table_to_tbar(ma_table)

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


def ki_and_variance_grid(ma_table, flux_on_readvar_pts):
    """Construct a grid of :math:`k` and covariances for the values of
    flux_on_readvar.

    The :math:`k` and corresponding covariances needed to do ramp fitting
    form essentially a one dimensional family in the flux in the ramp divided
    by the square of the read noise.  This function constructs these quantities
    for a large number of different flux / read_noise^2 to be used in
    interpolation.

    Parameters
    ----------
    ma_table : list[list] (int)
        a list of the first read and number of reads in each resultant
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
    cc_rn = construct_covar(1, 0, ma_table)
    cc_flux = construct_covar(0, 1, ma_table)
    outki = []
    outvar = []
    for flux_on_readvar in flux_on_readvar_pts:
        cc_flux_scaled = cc_flux * flux_on_readvar
        atcinva, atcinv = construct_ramp_fitting_matrices(
            cc_rn + cc_flux_scaled, ma_table)
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
    def __init__(self, ma_table, flux_on_readvar_pts=None):
        """Construct a RampFitInterpolator for an ma_table and a grid of
        flux/read_noise**2.

        Parameters
        ----------
        ma_table : list[list] (int)
            list of lists of first reads, number of reads in each resultant
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
        ki, var = ki_and_variance_grid(ma_table, self.flux_on_readvar_pts)
        self.ma_table = ma_table
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
        mingrid = self.flux_on_readvar_pts[0]
        maxgrid = self.flux_on_readvar_pts[-1]
        fluxonreadvar = flux / read_noise**2
        fluxonreadvar = np.clip(
            fluxonreadvar, self.flux_on_readvar_pts[0],
            self.flux_on_readvar_pts[-1])

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
        mingrid = self.flux_on_readvar_pts[0]
        maxgrid = self.flux_on_readvar_pts[-1]
        fluxonreadvar = flux / read_noise**2
        fluxonreadvar = np.clip(
            fluxonreadvar, self.flux_on_readvar_pts[0],
            self.flux_on_readvar_pts[-1])
        return self.var_interpolator(fluxonreadvar).astype('f4')*read_noise**2

    def fit_ramps(self, resultants, read_noise, fluxest=None):
        """Fit ramps for a set of resultants and their read noise.

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
        tbar = ma_table_to_tbar(self.ma_table)
        dtbar = np.diff(tbar).astype('f4')
        differences = resultants_to_differences(resultants)
        if fluxest is None:
            dtbar_reshape = dtbar.reshape(
                (dtbar.shape[0],) + (1,)*len(differences.shape[1:]))
            fluxest = np.median(differences[1:]/dtbar_reshape, axis=0)
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
def simulate_many_ramps(ntrial=100, flux=100, readnoise=5, ma_table=None):
    """Simulate many ramps with a particular flux, read noise, and ma_table.

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
    ma_table : list[list] (int)
        list of lists indicating first read and number of reads in each
        resultant

    Returns
    -------
    ma_table : list[list] (int)
        ma_table used
    flux : float
        flux used
    readnoise : float
        read noise used
    resultants : np.ndarray[n_resultant, ntrial] (float)
        simulated resultants
    """
    from . import l1
    read_time = parameters.read_time
    if ma_table is None:
        ma_table = [[1, 4], [5, 1], [6, 3], [9, 10], [19, 3], [22, 15]]
    nread = np.array([x[1] for x in ma_table])
    tij = l1.ma_table_to_tij(ma_table)
    totcounts = np.random.poisson(flux*np.max(np.concatenate(tij)), ntrial)
    resultants = l1.apportion_counts_to_resultants(totcounts, tij)
    resultants += np.random.randn(len(ma_table), ntrial)*(
        readnoise/np.sqrt(nread)).reshape(len(ma_table), 1)
    return (ma_table, flux, readnoise, resultants)


