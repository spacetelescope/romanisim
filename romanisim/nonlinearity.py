"""Routines to handle non-linearity in simulating ramps.

The approach taken here is straightforward.  The detector is accumulating
photons, but the capacitance of the pixel varies with flux level and so
the mapping between accumulated photons and read-out digital numbers
changes with flux level.  The CRDS linearity and inverse-linearity reference
files describe the mapping between linear DN and observed DN.  This module
implements that mapping.  When simulating an image, the photons entering
each pixel are simulated, and then before being "read out" into a buffer,
are transformed with this mapping into observed counts.  These are then
averaged and emitted as resultants.
"""

import numpy as np
from astropy import units as u
from romanisim import parameters


def repair_coefficients(coeffs, dq):
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

    dq : np.ndarray[n_resultant, nx, ny]
        Data Quality array

    Returns
    -------
    coeffs : np.ndarray[ncoeff, nx, ny] (float)
        "repaired" coefficients with NaNs and weird coefficients replaced with
        linear values with slopes of unity.

    dq : np.ndarray[n_resultant, nx, ny]
        DQ array marking pixels with improper non-linearity coefficients
    """
    res = coeffs.copy()

    nocorrection = np.zeros(coeffs.shape[0], dtype=coeffs.dtype)
    nocorrection[1] = 1.  # "no correction" is just normal linearity.
    # For NaN, all zero, or flagged pixels, reset to no linearity correction.
    m = (np.any(~np.isfinite(coeffs), axis=0) | np.all(coeffs == 0, axis=0)
         | (dq != 0))
    res[:, m] = nocorrection[:, None]

    lin_dq_array = np.zeros(coeffs.shape[1:], dtype=np.uint32)
    lin_dq_array[m] = parameters.dqbits['no_lin_corr']
    dq = np.bitwise_or(dq, lin_dq_array)
    return res, dq


def evaluate_nl_polynomial(counts, coeffs, reversed=False):
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

    if isinstance(counts, u.Quantity):
        unit = counts.unit
        counts = counts.value
    else:
        unit = None

    res = np.polyval(cc, counts)

    if unit is not None:
        res = res * unit

    return res


class NL:
    """Keep track of non-linearity and inverse non-linearity coefficients.

    """
    def __init__(self, coeffs, dq=None, gain=None):
        """Construct an NL class handling non-linearity correction.

        Parameters
        ----------
        coeffs : np.ndarray[ncoeff, nx, ny] (float)
            Non-linearity coefficients from reference files.

        dq : np.ndarray[n_resultant, nx, ny]
            Data Quality array

        gain : float or np.ndarray[float]
            Gain (electrons / count) for converting counts to electrons
        """
        if dq is None:
            dq = np.zeros(coeffs.shape[1:], dtype='uint32')
        if gain is None:
            gain = parameters.gain.to(u.electron / u.DN).value
        self.coeffs, self.dq = repair_coefficients(coeffs, dq)
        self.gain = gain

    def apply(self, counts, electrons=False, reversed=False):
        """Compute the correction of observed to true counts

        Parameters
        ----------
        counts : np.ndarray[nx, ny] (float)
            The observed counts

        electrons : bool
            Set to True for 'counts' being in electrons, with coefficients
            designed for DN. Accrdingly, the gain needs to be removed and
            reapplied.

        reversed : bool
            If True, the coefficients are in reversed order, which is the
            order that np.polyval wants them.  One can maybe save a little
            time reversing them once ahead of time.

        Returns
        -------
        corrected : np.ndarray[nx, ny] (float)
¯            The corrected counts.
        """
        if electrons:
            return self.gain * evaluate_nl_polynomial(counts / self.gain, self.coeffs, reversed)

        return evaluate_nl_polynomial(counts, self.coeffs, reversed)
