import numpy as np
cimport numpy as np
cimport cython
import romanisim.ramp
from libc.math cimport sqrt, fabs

# Casertano+2022, Table 2
cdef float[2][6] PTABLE = [
    [-np.inf, 5, 10, 20, 50, 100],
    [0,     0.4,  1,  3,  6,  10]]
cdef int PTABLE_LENGTH = 6


cdef inline float get_weight_power(float s):
    cdef int i
    for i in range(PTABLE_LENGTH):
        if s < PTABLE[0][i]:
            return PTABLE[1][i - 1]
    return PTABLE[1][i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.cpow(True)
cdef inline (float, float, float) fit_one_ramp(
        float [:] resultants, int start, int end, float read_noise,
        float [:] tbar, float [:] tau, int [:] nn):
    """Fit a portion of single ramp using the Casertano+22 algorithm.

    Parameters
    ----------
    resultants : float [:]
        array of resultants for single pixel
    start : int
        starting point of portion to fit within this pixel
    end : int
        ending point of portion to fit within this pixel
    read_noise : float
        read noise for this pixel
    tbar : float [:]
        mean times of resultants
    tau : float [:]
        variance weighted mean times of resultants
    nn : int [:]
        number of reads contributing to reach resultant

    Returns
    -------
    slope : float
        fit slope
    slopereadvar : float
        read noise induced variance in slope
    slopepoissonvar : float
        coefficient of Poisson-noise induced variance in slope
        multiply by true flux to get actual Poisson variance.
    """
    cdef int i = 0, j = 0
    cdef int nres = end - start + 1
    cdef float ww[2048]
    cdef float kk[2048]
    cdef float slope = 0, slopereadvar = 0, slopepoissonvar = 0
    cdef float tbarmid = (tbar[start] + tbar[end]) / 2

    # Doesn't make sense to fit a ramp with 1 or fewer resultant.  Fail early.
    if nres <= 1:
        return (0.0, 0.0, 0.0)

    # Casertano+2022 Eq. 44
    # Note we've departed from Casertano+22 slightly;
    # there s is just resultants[end].  But that doesn't seem good if, e.g.,
    # a CR in the first resultant has boosted the whole ramp high but there
    # is no actual signal.
    cdef float s = max(resultants[end] - resultants[start], 0)
    s = s / sqrt(read_noise**2 + s)
    cdef float weight_power = get_weight_power(s)

    # It's easy to use up a lot of dynamic range on something like
    # (tbar - tbarmid) ** 10.  Rescale these.
    cdef float tscale = (tbar[end] - tbar[start]) / 2
    if tscale == 0:
        tscale = 1
    cdef float f0 = 0, f1 = 0, f2 = 0
    for i in range(nres):
        # Casertano+22, Eq. 45
        ww[i] = ((((1 + weight_power) * nn[start + i]) /
            (1 + weight_power * nn[start + i])) *
            fabs((tbar[start + i] - tbarmid) / tscale) ** weight_power)
        # Casertano+22 Eq. 35
        f0 += ww[i]
        f1 += ww[i] * tbar[start + i]
        f2 += ww[i] * tbar[start + i]**2
    # Casertano+22 Eq. 36
    cdef float dd = f2 * f0 - f1 ** 2
    if dd == 0:
        return (0.0, 0.0, 0.0)
    for i in range(nres):
        # Casertano+22 Eq. 37
        kk[i] = (f0 * tbar[start + i] - f1) * ww[i] / dd
    for i in range(nres):
        # Casertano+22 Eq. 38
        slope += kk[i] * resultants[start + i]
        # Casertano+22 Eq. 39
        slopereadvar += kk[i] ** 2 * read_noise ** 2 / nn[start + i]
        # Casertano+22 Eq 40
        slopepoissonvar += kk[i] ** 2 * tau[start + i]
        for j in range(i + 1, nres):
            slopepoissonvar += 2 * kk[i] * kk[j] * tbar[start + i]

    return (slope, slopereadvar, slopepoissonvar)


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ramps(np.ndarray[float, ndim=2] resultants,
                np.ndarray[int, ndim=2] dq,
                np.ndarray[float, ndim=1] read_noise, ma_table):
    """Fit ramps using the Casertano+22 algorithm.

    This implementation fits all ramp segments between bad pixels
    marked in the dq image with values not equal to zero.  So the
    number of fit ramps can be larger than the number of pixels.
    The derived slopes, corresponding variances, and the locations of
    the ramps in each pixel are given in the returned dictionary.

    Parameters
    ----------
    resultants : np.ndarry[nresultants, npixel]
        the resultants in electrons
    dq : np.ndarry[nresultants, npixel]
        the dq array.  dq != 0 implies bad pixel / CR.
    read noise: float
        the read noise in electrons
    ma_table : list[list[int]]
        the ma table prescription

    Returns
    -------
    dictionary containing the following keywords:
    slope : np.ndarray[nramp]
        slopes fit for each ramp
    slopereadvar : np.ndarray[nramp]
        variance in slope due to read noise
    slopepoissonvar : np.ndarray[nramp]
        variance in slope due to Poisson noise, divided by the slope
        i.e., the slope poisson variance is coefficient * flux; this term
        is the coefficient.
    pix : np.ndarray[nramp]
        the pixel each ramp is in
    resstart : np.ndarray[nramp]
        The first resultant in this ramp
    resend : np.ndarray[nramp]
        The last resultant in this ramp.
    """
    cdef int nresultant = len(ma_table)
    cdef np.ndarray[int] nn = np.array([x[1] for x in ma_table]).astype('i4')
    # number of reads in each resultant
    cdef np.ndarray[float] tbar = romanisim.ramp.ma_table_to_tbar(
        ma_table).astype('f4')
    cdef np.ndarray[float] tau = romanisim.ramp.ma_table_to_tau(
        ma_table).astype('f4')
    cdef int npixel = resultants.shape[1]
    cdef int nramp = (np.sum(dq[0, :] == 0) +
                      np.sum((dq[:-1, :] != 0) & (dq[1:, :] == 0)))
    cdef np.ndarray[float] slope = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[float] slopereadvar = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[float] slopepoissonvar = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[int] resstart = np.zeros(nramp, dtype='i4') - 1
    cdef np.ndarray[int] resend = np.zeros(nramp, dtype='i4') - 1
    cdef np.ndarray[int] pix = np.zeros(nramp, dtype='i4') - 1
    cdef int i, j
    cdef int inramp = -1
    cdef int rampnum = 0
    for i in range(npixel):
        inramp = 0
        for j in range(nresultant):
            if (not inramp) and (dq[j, i] == 0):
                inramp = 1
                pix[rampnum] = i
                resstart[rampnum] = j
            elif (not inramp) and (dq[j, i] != 0):
                continue
            elif inramp and (dq[j, i] == 0):
                continue
            elif inramp and (dq[j, i] != 0):
                inramp = 0
                resend[rampnum] = j - 1
                rampnum += 1
            else:
                raise ValueError('unhandled case')
        if inramp:
            resend[rampnum] = j
            rampnum += 1
    # we should have just filled out the starting and stopping locations
    # of each ramp.

    cdef float slope0, slopereadvar0, slopepoissonvar0
    cdef float [:, :] resview = resultants
    cdef float [:] rnview = read_noise
    cdef float [:] tbarview = tbar
    cdef float [:] tauview = tau
    cdef int [:] nnview = nn
    for i in range(nramp):
        slope0, slopereadvar0, slopepoissonvar0 = fit_one_ramp(
            resview[:, pix[i]], resstart[i], resend[i], rnview[pix[i]],
            tbarview, tauview, nnview)
        slope[i] = slope0
        slopereadvar[i] = slopereadvar0
        slopepoissonvar[i] = slopepoissonvar0

    return dict(slope=slope, slopereadvar=slopereadvar,
                slopepoissonvar=slopepoissonvar,
                pix=pix, resstart=resstart, resend=resend)
