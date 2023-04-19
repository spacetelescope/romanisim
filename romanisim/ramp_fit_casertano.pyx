import numpy as np
cimport numpy as np
cimport cython
import romanisim.ramp

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ramps(np.ndarray[float, ndim=2] resultants,
              np.ndarray[int, ndim=2] dq,
              np.ndarray[float, ndim=1] read_noise, ma_table):
    """Fit ramps using the Casertano+ algorithm.

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
    # Casertano+2022, Table 2
    cdef np.ndarray[float, ndim=2] ptable = np.array([
        [-np.inf, 0], [5, 0.4], [10, 1], [20, 3], [50, 6], [100, 10]]
	).astype('f4')
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

    cdef np.ndarray[float] smax = resultants[resend, pix]
    # Casertano+2022 Eq. 44
    cdef np.ndarray[float] s = smax / np.sqrt(read_noise[pix]**2 + smax)
    cdef np.ndarray[float] pp = ptable[np.searchsorted(ptable[:, 0], s) - 1, 1]
    cdef np.ndarray[float] tbarmid = (tbar[resstart] + tbar[resend]) / 2

    # Casertano+22, Eq. 45
    cdef np.ndarray[float, ndim=2] ww = np.zeros(
        (resultants.shape[0], resultants.shape[1]), dtype='f4')
    for i in range(nramp):
        for j in range(resstart[i], resend[i] + 1):
            ww[j, pix[i]] = (((1 + pp[i]) * nn[j]) / (1 + pp[i] * nn[j])
	                     * abs(tbar[j] - tbarmid[i]) ** pp[i])

    cdef np.ndarray[float] f0 = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[float] f1 = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[float] f2 = np.zeros(nramp, dtype='f4')
    for i in range(nramp):
        for j in range(resstart[i], resend[i] + 1):
            # Casertano+22 Eq. 35
            f0[i] += ww[j, pix[i]]
            f1[i] += ww[j, pix[i]] * tbar[j]
            f2[i] += ww[j, pix[i]] * tbar[j]**2
    # Casertano+22 Eq. 36
    cdef np.ndarray[float] dd = f2 * f0 - f1**2

    cdef int bad = 0
    for i in range(nramp):
        for j in range(resstart[i], resend[i] + 1):
            # Casertano+22 Eq. 37
            # Note that we are replacing ww with kk to save memory; we don't
            # need ww again.
            bad = dd[i] == 0
            ww[j, pix[i]] = ((f0[i] * tbar[j] - f1[i]) * ww[j, pix[i]] /
                             (dd[i] + bad))
            ww[j, pix[i]] *= (bad == 0)  # zero weights for bad ramps.

    for i in range(nramp):
        for j in range(resstart[i], resend[i] + 1):
            # Casertano+22 Eq. 38
            slope[i] += ww[j, pix[i]] * resultants[j, pix[i]]
            # Casertano+22 Eq. 39
            slopereadvar[i] += ww[j, pix[i]]**2 * read_noise[pix[i]]**2 / nn[j]
            # Casertano+22 Eq 40
            slopepoissonvar[i] += ww[j, pix[i]]**2 * tau[j]
            for m in range(j + 1, resend[i] + 1):
                slopepoissonvar[i] += (2 * ww[j, pix[i]]
		                       * ww[m, pix[i]] * tbar[j])

    # let's just return the bare minimum from the cython code and let the caller
    # clean up.
    return dict(slope=slope, slopereadvar=slopereadvar,
                slopepoissonvar=slopepoissonvar,
                pix=pix, resstart=resstart, resend=resend)
