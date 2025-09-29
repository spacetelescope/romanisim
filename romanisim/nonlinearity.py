"""Routines to handle non-linearity in simulating ramps.

The approach taken here is straightforward.  The detector is accumulating
electrons, but the capacitance of the pixel varies with flux level and so
the mapping between accumulated electrons and read-out digital numbers
changes with flux level.  The CRDS linearity and inverse-linearity reference
files describe the mapping between linear DN and observed DN.  This module
implements that mapping.  When simulating an image, the electrons entering
each pixel are simulated, and then before being "read out" into a buffer,
are transformed with this mapping into observed electrons.  These are then
averaged and emitted as resultants.

Note that there is an approximation happening here surrounding
the treatment of electrons vs. DN.  During the simulation of the individual
reads, all operations, including linearity, work in electrons.  Nevertheless
we apply non-linearity at this time, transforming electrons into "non-linear"
electrons using this module, which will be proportional to the final DN.  Later
in the L1 simulation these "non-linear" electrons are divided by the gain to
construct final DN image.
"""

import numpy as np
from astropy import units as u
from romanisim import parameters, log


def repair_coefficients(coeffs, dq):
    """Fix cases of zeros and NaNs in non-linearity coefficients.

    This function replaces suspicious-looking non-linearity coefficients with
    identity transformation coefficients from a non-linearity perspective; all
    coefficients are zero except for the linear term, which is set to 1.

    This function doesn't try to make sure that the derivative of the
    correction is greater than 1, which we would expect for a non-linearity
    correction.

    Parameters
    ----------
    coeffs : np.ndarray[ncoeff, ny, nx] (float)
        Nonlinearity coefficients, starting with the constant term and
        increasing in power.

    dq : np.ndarray[n_resultant, ny, nx]
        Data Quality array

    Returns
    -------
    coeffs : np.ndarray[ncoeff, ny, nx] (float)
        "repaired" coefficients with NaNs and weird coefficients replaced with
        linear values with slopes of unity.

    dq : np.ndarray[n_resultant, ny, nx]
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
    """Correct the observed DN for non-linearity.

    As electrons accumulate, they make it harder for the device to count
    future electrons due to classical non-linearity.  This function
    converts observed DN to what would have been seen absent
    non-linearity, using the provided non-linearity coefficients.

    Parameters
    ----------
    counts : np.ndarray[ny, nx] (float)
        Number of DN already in pixel
    coeffs : np.ndarray[ncoeff, ny, nx] (float)
        Coefficients of the non-linearity correction polynomials
    reversed : bool
        If True, the coefficients are in reversed order, which is the
        order that np.polyval wants them.  One can maybe save a little
        time reversing them once ahead of time.

    Returns
    -------
    corrected : np.ndarray[nx, ny] (float)
        The corrected number of DN
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
    def __init__(self, coeffs, dq=None, gain=None, saturation=None, inverse=False):
        """Construct an NL class handling non-linearity correction.

        Parameters
        ----------
        coeffs : np.ndarray[ncoeff, nx, ny] (float)
            Non-linearity coefficients from reference files.

        dq : np.ndarray[n_resultant, nx, ny]
            Data Quality array

        gain : float or np.ndarray[float]
            Gain (electrons / DN) for converting DN to electrons

        saturation : float or None
            Saturation level in DN

        inverse: bool
            True if this corresponds to the inverse linearity correction.

            This changes the interpretation of the saturation keyword, which is
            always the saturation level in observed DN.  This gets translated
            internally to linearized DN if inverse is True.
        """
        if dq is None:
            dq = np.zeros(coeffs.shape[1:], dtype='uint32')
        if gain is None:
            gain = parameters.reference_data['gain'].to(u.electron / u.DN).value

        self.coeffs, self.dq = repair_coefficients(coeffs, dq)
        self.gain = gain

        if saturation is not None and inverse:
            new_saturation = evaluate_nl_polynomial(saturation, self.coeffs)
            m = (new_saturation < 0 * u.DN) | (new_saturation > saturation * 1.5)
            if np.any(m):
                log.warning(
                    f'{np.sum(m)} points with problematic saturation / inverse linearity '
                    'values; setting saturation of these points to 10 DN!')
            new_saturation[m] = 10 * u.DN
            saturation = new_saturation

        self.saturation = saturation

    def apply(self, counts, electrons=False, reversed=False):
        """Compute the correction of DN to linearized DN.

        Alternatively, when electrons = True, rescale these to DN,
        correct the DN, and scale them back to electrons using
        the gain.

        Parameters
        ----------
        counts : np.ndarray[nx, ny] (float)
            The observed counts

        electrons : bool
            Set to True for 'counts' being in electrons, with coefficients
            designed for DN. Accordingly, the gain needs to be removed and
            reapplied.

        reversed : bool
            If True, the coefficients are in reversed order, which is the
            order that np.polyval wants them.  One can maybe save a little
            time reversing them once ahead of time.

        Returns
        -------
        corrected : np.ndarray[nx, ny] (float)
            The corrected DN or electrons.
        """

        gain = self.gain

        if electrons:
            if not isinstance(counts, u.Quantity):
                gain = gain / u.electron
            counts = counts / gain

        if self.saturation is not None:
            counts = np.clip(counts, -1000 * u.DN, self.saturation)

        corrected = evaluate_nl_polynomial(counts, self.coeffs, reversed)

        if electrons:
            corrected = corrected * gain

        return corrected
