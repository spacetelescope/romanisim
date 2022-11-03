"""Routines to handle non-linearity in simulating ramps.

As we sample up a ramp, we become decreasingly sensitive to light; given a
certain number of photons that we would have detected absent nonlinearity,
we instead detect a smaller number.  In simulations, we want the fraction
of the number of photons we detect relative to the number we would detect
absent nonlinearity.  Non-linearity is typically described instead by some
polynomial that takes the observed number of photons to the corrected number
of photons.

.. math:: C = f(O) \\, ,

where :math:`f` is a polynomial depending on some coefficients that are
different for each pixel, :math:`C` is the corrected number of photons, and
:math:`O` is the observed number of photons.

We want instead the current fraction of photons that would be detected in the
next read.  That's

.. math:: dO/dC = (df/dO)^{-1}

In the context of the up-the-ramp samples performed in the L1 simulations,
we need to reduce the fraction of photons selected from the binomial
distribution by this fraction in each read, and then adjust the fraction
of photons read in in future samples down further to reflect that the
photons lost to non-linearity will not be read out later down the ramp.

We do some simple sanity checks on the non-linearity coefficients, but we
do not yet check to make sure that :math:`dO/dC \\leq 1`.  i.e., we don't yet
check that the non-linearity correction function corresponds to a curve that
always means that the number of photons recorded is smaller than the number
of photons entering the read.  Such cases are likely to cause problems when
we try to sample more photons from the ramp than were present in the ramp.

The implementation of nonlinearity makes an important simplification:
that the efficiency is constant within a read.  This isn't quite true.  The
original implementation in `l1.py` also assumed that the efficiency was equal
to its value at the start of the read.  Since the efficiency is usually
dropping as the pixel fills up this leads to the simulation leaving in somewhat
more photons than expected.  For a particular simulated pixel that I inspected,
this was a 1% effect at counts = 0 to a 2% effect at counts = 100k, with most
of the range having a <0.5% effect.  To mitigate this effect, `l1.py` now
evaluates the efficiency at what it expects the efficiency to be at the middle
of the read, based on the previous efficiency and the length of the read.  This
adjustment reduces the bias to <5e-4 across the full range.  The next step would
be to take this another iteration further, but that feels like overkill.
"""

import numpy as np


def repair_coefficients(coeffs):
    """Fix cases of zeros and NaNs in non-linearity coefficients.

    This function replaces suspicious-looking non-linearity coefficients with
    no-op coefficients from a non-linearity perspective; all coefficients are
    zero except for the linear term, which is set to 1.

    This function doesn't try to make sure that the derivative of the
    correction is greater than 1, which we would expect for a non-linearity
    correction.

    Parameters
    ----------
    coeffs : np.ndarray[ncoeff, nx, ny] (float)
        Nonlinearity coefficients, starting with the constant term and
        increasing in power.

    Returns
    -------
    coeffs : np.ndarray[ncoeff, nx, ny] (float)
        "repaired" coefficients with NaNs and weird coefficients replaced with
        linear values with slopes of unity.
    """
    res = coeffs.copy()
    nocorrection = np.zeros(coeffs.shape[0], dtype=coeffs.dtype)
    nocorrection[1] = 1.  # "no correction" is just normal linearity.
    m = np.any(~np.isfinite(coeffs), axis=0) | np.all(coeffs == 0, axis=0)
    res[:, m] = nocorrection[:, None]
    return res


def derivative(coeffs):
    """Compute the coefficients of the derivative of non-linearity corrections.

    The non-linearity corrections are a polynomial; their derivatives are also
    a polynomial.  This computes those coefficients.

    Parameters
    ----------
    coeffs : np.ndarray[ncoeff, nx, ny] (float)
        Nonlinearity coefficients, starting with the constant term and
        increasing in power.

    Returns
    -------
    derivcoeffs : np.ndarray[ncoeff, nx, ny] (float)
        coefficients of the polynomial that is the derivative of the
        non-linearity correction polynomial
    """
    if np.any(~np.isfinite(coeffs)):
        raise ValueError('NaNs in nonlinearity coefficients.')
    if np.any(np.all(coeffs == 0, axis=0)):
        raise ValueError('All zero nonlinearity coefficients.')

    coeffs = np.array(coeffs)
    derivcoeffs = coeffs * np.arange(0, coeffs.shape[0]).reshape(
        coeffs.shape[0:1] + (1,) * (len(coeffs.shape) - 1))
    return derivcoeffs[1:, ...]


def efficiency(counts, derivcoeffs, reversed=False):
    """Convert the (in)efficiency of counting photons due to nonlinearity.

    As photons arrive, they make it harder for the device to count future
    photons due to classical non-linearity.  The "efficiency" here is the
    fraction of photons we should count.  It is the reciprocal of the
    derivative of the correction polynomial.

    Parameters
    ----------
    counts : np.ndarray[nx, ny] (float)
        Number of counts already in pixel
    derivcoeffs : np.ndarray[ncoeff, nx, ny] (float)
        Coefficients of the derivative of the non-linearity correction
        polynomial (e.g., from romanisim.nonlinearity.derivative).
    reversed : bool
        If True, the coefficients are in reversed order, which is the
        order that np.polyval wants them.  One can maybe save a little
        time reversing them once ahead of time.

    Returns
    -------
    efficiencies : np.ndarray[nx, ny] (float)
        The (instantaneous) fraction of arriving photons that will be counted.
    """
    if reversed:
        cc = derivcoeffs
    else:
        cc = derivcoeffs[::-1, ...]
    dfdo = np.polyval(cc, counts)
    return 1 / dfdo


def correct(counts, coeffs, reversed=False):
    """Correct the observed counts for non-linearity.

    As photons arrive, they make it harder for the device to count
    future photons due to classical non-linearity.  This function
    converts some observed counts to what would have been seen absent
    non-linearity given some non-linearity corrections described by
    polynomials with given coefficients.

    Parameters
    ----------
    counts : np.ndarray[nx, ny] (float)
        Number of counts already in pixel
    coeffs : np.ndarray[ncoeff, nx, ny] (float)
        Coefficients of the non-linearity correction polynomials
    reversed : bool
        If True, the coefficients are in reversed order, which is the
        order that np.polyval wants them.  One can maybe save a little
        time reversing them once ahead of time.

    Returns
    -------
    corrected : np.ndarray[nx, ny] (float)
        The corrected number of counts
    """
    if reversed:
        cc = coeffs
    else:
        cc = coeffs[::-1, ...]
    return np.polyval(cc, counts)


class NL:
    """Keep track of non-linearity coefficients.

    This is a wrapper class to help other classes need to know less
    about non-linearity, but it doesn't presently do much more than
    cache the non-linearity derivative coefficients so that they may be
    used multiple times in the up-the-ramp sampling.

    It would be nice to encapsulate more information about the loss of
    photons up the read that is presently in `l1.py`, but that's pretty
    tightly coupled to the apportionment mechanism and so I haven't
    explored that adequately.
    """
    def __init__(self, coeffs):
        """Construct an NL class handling non-linearity correction.

        Parameters
        ----------
        coeffs : np.ndarray[ncoeff, nx, ny] (float)
            Non-linearity coefficients from reference files.
        """
        self.coeffs = repair_coefficients(coeffs)
        self.derivcoeffs = derivative(self.coeffs)
        self.coeffs = self.coeffs[::-1, ...].copy()
        self.derivcoeffs = self.derivcoeffs[::-1, ...].copy()

    def efficiency(self, counts):
        """Compute the efficiency of photon counting given a count level.

        Parameters
        ----------
        counts : np.ndarray[nx, ny] (float)
            The counts in each pixel

        Returns
        -------
        efficiency : np.ndarray[nx, ny] (float)
            The efficiency of each pixel at the current count level.
        """
        return efficiency(counts, self.derivcoeffs, reversed=True)

    def correct(self, counts):
        """Compute the correction of observed to true counts

        Parameters
        ----------
        counts : np.ndarray[nx, ny] (float)
            The observed counts

        Returns
        -------
        corrected : np.ndarray[nx, ny] (float)
            The corrected counts.
        """
        return correct(counts, self.coeffs, reversed=True)
