"""This module contains routines that convert total electron images into L1 images.

We imagine starting with an image that gives the total number of electrons from
all Poisson processes (at least: sky, sources, dark current).  One then needs
to redistribute these electrons over the resultants of an L1 image.

The remainder of this module-level discussion covers some of the philosophical
choices taken in this module and the consequences thereof.  It is more
colloquial than much of the documentantion and may be ignored.

The easiest approach is to sample the
image read-by-read with a binomial draw from the total counts weighted by the
chance that each count landed in this particular time window, and then gather
those for each resultant and average, as done on the spacecraft.

It's tempting to go straight to making the appropriate resultants.  Following
Casertano (2022), the variance in each resultant is:

.. math:: V = \\sigma_{read}^2/N + f \\tau

where f is the count rate, N is the number of reads in the resultant, and :math:`\\tau`
is the 'variance-based resultant time'

.. math:: \\tau = 1/N^2 \\sum_{reads} (2 (N - k) - 1) t_k

where the t_k is the time of the kth read in the resultant.

For uniformly spaced reads,

.. math:: \\tau = t_0 + d (N/3 + 1/6N - 1/2) \\, ,

where t_0 is the time of the first read in the resultant and d is the spacing
of the reads.

That gives the variance from Poisson counts in the resultants.  But how should
we draw random numbers to get that variance and the right mean?  One can
separately control the mean and variance by scaling the Poisson distribution,
but that will not do the the right thing with the higher order moments.

It isn't that expensive to just work out all of the reads,
individually, which also allows more natural incorporation of
cosmic rays.  So we take that approach.

What is the input for constructing an L1 image?  An L1 image is defined by a
total electron image and a list of lists :math:`t_{i, j}`, where
:math:`t_{i, j}` is the time at which the jth read in the ith
resultant is made.  We demand :math:`t_{i, j} > t_{k, l}`
whenever i > k or whenever i = k and j > l.

This approach doesn't allow for some effects, for example:

* jitter in telescope pointing: the rate image is the same for each read/resultant
* weird non-linear systematics in darks?

Some systematics need to be applied to the individual reads, rather than to
the final image.  Currently linearity, persistence, and CRs are implemented at
individual read time.

This approach is not super fast.  For a high latitude set of resultants,
generating all of the random numbers to determine the apportionment takes
43 s on a 2020 Mac laptop; this will scale linearly with the
number of reads.  That's longer than the actual image production for the
dummy scene I'm using (though only ~2x longer).

It's unclear how to speed this up.  Explicitly doing the Poisson
noise on each read from a rate image (rather than the apportionment of an
image that already has Poisson noise) is 2x slower---generating billions
of random numbers just takes a little while.

Plausibly one could figure out how to draw numbers directly from what a
resultant is rather than looking at each read individually.  That would
likely bring a ~10x speed-up.  The read noise there is easy.  The
Poisson noise is a sum of scaled Poisson variables:

.. math:: \\sum_{i=0, ..., N-1} (N-i) c_i \\, ,

where :math:`c_i` is a Poisson-distributed variable.
The sum of Poisson-distributed variables is Poisson-distributed, but I wasn't
able to find an easy distribution to use for the sum of scaled Poisson-distributed
quantities.

If one did sample from that directly, we'd still also need to get the sum
of the counts in the reads comprising the resultant.  So one would need
a separate draw for that, conditional on the average obtained for the resultant.
Or, reversing that, one could draw the total number of counts first,
e.g., via the binomial distribution, and then would draw the number
for what the average number of counts was among the reads comprising the
resultant, conditional on the total number of counts.  Then

.. math:: \\sum_{i=0, ..., N-1} (N-i) c_i

is some kind of statistic of the multinomial distribution.  That looks more tractable:

.. math:: c_i \\sim \\mathrm{multinomial}(\\mathrm{total}, [1/N, ..., 1/N])

We need to draw from :math:`\\sum (N-i) c_i`.
The probabilities are always :math:`1/N`, with the possible small but
important
exception of 'skipped' or 'dropped' reads, in which case the first read would
be more like :math:`2/(N+1)` and all the others :math:`1/(N+1)`.  If the
probabilities were always 1/N, this looks vaguely like it could have a nice analytic
solution.  We don't pursue this avenue further here.
"""

import numpy as np
import asdf
import galsim
from scipy import ndimage
from . import parameters
from . import log
from astropy import units as u
from . import cr


def validate_times(tij):
    """Verify that a set of times tij is ascending.

    Parameters
    ----------
    tij : list[list[float]]
        a list of lists of times at which each read in a resultant is performed

    Returns
    -------
    bool
        True if the tij are ascending, otherwise False
    """
    times = [t for resultant in tij for t in resultant]
    return np.all(np.diff(times) > 0)


def tij_to_pij(tij, remaining=False):
    """Convert a set of times tij to corresponding probabilities for sampling.

    The probabilities are those needed for sampling from a binomial
    distribution for each read.  These are delta_t / sum(delta_t), the
    fraction of time in each read over the total time, when `remaining` is
    False.  When remaining is true, these probabilities are scaled not by
    the total time but by the remaining time, so that subsequent reads get
    subsequent reads get scaled up so that each pij is
    delta_t / time_remaining, and the last read always has pij = 1.

    Parameters
    ----------
    tij : list[list[float]]
        list of lists of readout times for each read entering a resultant
    remaining : bool
        scale by remaining time rather than total time

    Returns
    -------
    list[list[float]]
        list of list of probabilities for each read, corresponding to the
        chance that a photon not yet assigned to a read so far should be
        assigned to this read.
    """
    if not validate_times(tij):
        raise ValueError('The given tij are not valid ascending resultant '
                         'times!')
    texp = tij[-1][-1]  # total exposure time
    tremaining = texp
    pij = []
    tlast = 0
    for resultant in tij:
        pi = []
        for t in resultant:
            pi.append(min([(t - tlast) / tremaining, 1]))
            if remaining:
                tremaining -= (t - tlast)
            tlast = t
        pij.append(pi)
    return pij


def apportion_counts_to_resultants(
        counts, tij, inv_linearity=None, crparam=None, persistence=None,
        tstart=None, rng=None, seed=None):
    """Apportion counts to resultants given read times.

    This finds a statistically appropriate assignment of electrons to each
    read composing a resultant, and averages the reads together to make
    the resultants.

    We loop over the reads, each time sampling from the counts image according
    to the probability that a photon lands in that particular read.  This
    is implemented via np.random.binomial(number of counts left, p/p_left) .

    We then average the reads together to get a resultant.

    We accumulate:

    * a sum for the resultant, which is divided by the number of reads and
      returned in the resultants array
    * a sum for the total number of photons accumulated so far, so we know
      where to start the next resultant
    * the resultants so far

    Parameters
    ----------
    counts : np.ndarray[ny, nx] (int)
        The number of counts in each pixel from sources in the final image
        This final image should be a ~conceptual image of the scene observed
        by an idealized instrument seeing only backgrounds and sources and
        observing until the end of the last read; no instrumental effects are
        included beyond PSF & distortion.
    tij : list[list[float]]
        list of lists of readout times for each read entering a resultant
    inv_linearity : romanisim.nonlinearity.NL or None
        Object implementing inverse non-linearity correction
    crparam : dict
        Dictionary of keywords sent to romanisim.cr.simulate_crs for simulating
        cosmic rays.  If None, no CRs are added
    persistence : romanisim.persistence.Persistence or None
        Persistence object describing persistence-affected photons, or None
        if persistence should not be simulated.
    tstart : astropy.time.Time
        Time of exposure start.  Used only if persistence is not None.
    rng : galsim.BaseDeviate
        random number generator
    seed : int
        seed to use for random number generator

    Returns
    -------
    resultants, dq
    resultants : np.ndarray[n_resultant, nx, ny]
        array of n_resultant images giving each resultant
    dq : np.ndarray[n_resultant, nx, ny]
        dq array marking CR hits in resultants
    """
    if not np.all(counts == np.round(counts)):
        raise ValueError('apportion_counts_to_resultants expects the counts '
                         'to be integers!')
    counts = np.clip(counts, 0, 2 * 10**9).astype('i4')

    # Set rng for creating cosmic rays, persistence, and readnoise
    if rng is None and seed is None:
        seed = 46
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.GaussianDeviate(seed)

    rng_numpy_seed = rng.raw()
    rng_numpy = np.random.default_rng(rng_numpy_seed)
    rng_numpy_cr = np.random.default_rng(rng_numpy_seed + 1)
    rng_numpy_ps = np.random.default_rng(rng_numpy_seed + 2)
    # two separate generators so that if you turn off CRs / persistence
    # you don't change the image

    # Convert readout times for each read entering a resultant to probabilities,
    # corresponding to the chance that a photon not yet assigned to a read so far
    # should be assigned to this read.
    pij = tij_to_pij(tij, remaining=True)

    # Create arrays to store various photon or electron counts and dq
    resultants = np.zeros((len(tij),) + counts.shape, dtype='f4')
    counts_so_far = np.zeros(counts.shape, dtype='i4')
    resultant_counts = np.zeros(counts.shape, dtype='f4')
    dq = np.zeros(resultants.shape, dtype=np.uint32)

    # Set initial instrument counts
    instrumental_so_far = 0

    if crparam is not None or persistence is not None:
        instrumental_so_far = np.zeros(counts.shape, dtype='i4')

    if persistence is not None and tstart is None:
        raise ValueError('tstart must be set if persistence is set!')

    if persistence is not None:
        tstart = tstart.mjd

    # Loop over read probabilities
    for i, pi in enumerate(pij):
        # Reset resultant counts
        resultant_counts[...] = 0

        # Loop over resultant probabilities
        for j, p in enumerate(pi):
            # Set read counts
            read = rng_numpy.binomial(counts - counts_so_far, p)
            counts_so_far += read

            # Apply cosmic rays
            if crparam is not None:
                old_instrumental_so_far = instrumental_so_far.copy()
                cr.simulate_crs(instrumental_so_far, parameters.read_time,
                                **crparam, rng=rng_numpy_cr)
                crhits = instrumental_so_far != old_instrumental_so_far
                dq[i, crhits] |= parameters.dqbits['jump_det']

            # Apply persistence
            if persistence is not None:
                tnow = tstart + tij[i][j] / (24 * 60 * 60)
                persistence.add_to_read(
                    instrumental_so_far, tnow, rng=rng_numpy_ps)

            # Update counts for the resultant
            if inv_linearity is not None:
                # Apply inverse linearity
                resultant_counts += inv_linearity.apply(
                    counts_so_far + instrumental_so_far, electrons=True)
            else:
                resultant_counts += counts_so_far + instrumental_so_far

        # set the read count to the average of the resultant count
        resultants[i, ...] = resultant_counts / len(pi)

    if inv_linearity is not None:
        # Update data quality array for inverse linearty coefficients
        dq |= inv_linearity.dq

    if persistence is not None:
        # should the electrons from persistence contribute to future
        # persistence?  Here they do.  But hopefully this choice is second
        # order enough that either decision would be fine.
        persistence.update(counts_so_far + instrumental_so_far, tnow)

    return resultants, dq


def add_read_noise_to_resultants(resultants, tij, read_noise=None, rng=None,
                                 seed=None, pedestal_extra_noise=None):
    """Adds read noise to resultants.

    The resultants get Gaussian read noise with sigma = sigma_read/sqrt(N).
    This is not quite right.  In reality read noise is added during each read.
    This is the same as adding to the resultants and dividing by sqrt(N) except
    for quantization; this additional subtlety is currently ignored.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, ny, nx] (float)
        resultants array, giving each of n_resultant resultant images
    tij : list[list[float]]
        list of lists of readout times for each read entering a resultant
    read_noise : float or np.ndarray[ny, nx] (float)
        read noise or read noise image for adding to resultants
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.
    pedestal_extra_noise : float
        Extra read noise to add to each pixel across all groups.
        Equivalent to noise in the reference read.

    Returns
    -------
    np.ndarray[n_resultant, ny, nx] (float)
        resultants with added read noise
    """
    if rng is None and seed is None:
        seed = 45
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.GaussianDeviate(seed)
    else:
        rng = galsim.GaussianDeviate(rng)

    # separate noise generator for pedestals so we can turn it on and off.
    pedestalrng = galsim.GaussianDeviate(rng.raw())

    if read_noise is None:
        read_noise = parameters.reference_data['readnoise']
    if read_noise is None:
        log.warning('Not applying read noise due to weird reference data.')
        return resultants

    noise = np.zeros(resultants.shape, dtype='f4')
    rng.generate(noise)
    noise = noise * read_noise / np.array(
        [len(x)**0.5 for x in tij]).reshape(-1, 1, 1)
    resultants += noise

    if pedestal_extra_noise is not None:
        noise = np.zeros(resultants.shape[1:], dtype='f4')
        amplitude = np.hypot(pedestal_extra_noise, read_noise)
        pedestalrng.generate(noise)
        resultants += noise[None, ...] * amplitude

    return resultants


def make_asdf(resultants, dq=None, filepath=None, metadata=None, persistence=None):
    """Package and optionally write out an L1 frame.

    This routine packages an L1 data file with the appropriate Roman data
    model.  It currently does not do anything with the necessary metadata,
    and leaves that information as filler values.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, ny, nx] (float)
        resultants array, giving each of n_resultant resultant images
    filepath : str
        if not None, path of asdf file to L1 image into
    dq : np.ndarray[n_resultant, ny, nx] (int)
        dq array flagging saturated / CR hit pixels

    Returns
    -------
    roman_datamodels.datamodels.ScienceRawModel
        L1 image
    extras : dict
        dictionary of additionally tabulated quantities, potentially
        including DQ images and persistence information.
    """

    from roman_datamodels import stnode
    nborder = parameters.nborder
    npix = galsim.roman.n_pix + 2 * nborder
    out = stnode.WfiScienceRaw.fake_data(shape=(len(resultants), npix, npix))
    if metadata is not None:
        out['meta'].update(metadata)
    extras = dict()
    out['data'][:, nborder:-nborder, nborder:-nborder] = resultants.value
    if dq is not None:
        extras['dq'] = np.zeros(out['data'].shape, dtype='i4')
        extras['dq'][:, nborder:-nborder, nborder:-nborder] = dq
    if persistence is not None:
        extras['persistence'] = persistence.to_dict()
    if filepath:
        af = asdf.AsdfFile()
        af.tree = {'roman': out, 'romanisim': extras}
        af.write_to(filepath)
    return out, extras


def read_pattern_to_tij(read_pattern):
    """Get the times of each read going into resultants for a read_pattern.

    Parameters
    ----------
    read_pattern : int or list[list]
        If int, id of ma_table to use.
        Otherwise a list of lists giving the indices of the reads entering each
        resultant.

    Returns
    -------
    list[list[float]]
        list of lists of readout times for each read entering a resultant
    """
    if isinstance(read_pattern, int):
        read_pattern = parameters.read_pattern[read_pattern]
    tij = [parameters.read_time * np.array(x) for x in read_pattern]
    return tij


def add_ipc(resultants, ipc_kernel=None):
    """Add IPC to resultants.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, ny, nx]
        resultants describing a scene

    Returns
    -------
    np.ndarray[n_resultant, ny, nx]
        resultants with IPC
    """
    # add in IPC
    # the reference pixels have basically no flux, so for these real pixels we
    # extend the array with a constant equal to zero.
    if ipc_kernel is None:
        ipc_kernel = parameters.ipc_kernel

    log.info('Adding IPC...')
    out = ndimage.convolve(resultants, ipc_kernel[None, ...],
                           mode='constant', cval=0)
    return out


def make_l1(counts, read_pattern,
            read_noise=None, pedestal_extra_noise=None,
            rng=None, seed=None,
            gain=None, inv_linearity=None, crparam=None,
            persistence=None, tstart=None, saturation=None):
    """Make an L1 image from a total electrons image.

    This apportions the total electrons among the different resultants and adds
    some instrumental effects (linearity, IPC, CRs, persistence, ...).

    Parameters
    ----------
    counts : galsim.Image
        total electrons delivered to each pixel
    read_pattern : int or list[list]
        MA table number or list of lists giving indices of reads entering each
        resultant.
    read_noise : np.ndarray[ny, nx] (float) or float
        Read noise entering into each read
    pedestal_extra_noise : np.ndarray[ny, nx] (float) or float
        Extra noise entering into each pixel (i.e., degenerate with pedestal)
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.
    gain : float or np.ndarray[float]
        Gain (electrons / DN) for converting DN to electrons
    inv_linearity : romanisim.nonlinearity.NL or None
        Object describing the inverse non-linearity corrections.
    crparam : dict
        Keyword arguments to romanisim.cr.simulate_crs.  If None, no
        cosmic rays are simulated.
    persistence : romanisim.persistence.Persistence
        Persistence instance describing persistence-affected pixels
    tstart : astropy.time.Time
        time of exposure start

    Returns
    -------
    l1, dq
    l1: np.ndarray[n_resultant, ny, nx]
        resultants image array including systematic effects
    dq: np.ndarray[n_resultant, ny, nx]
        DQ array marking saturated pixels and cosmic rays
    """

    tij = read_pattern_to_tij(read_pattern)
    log.info('Apportioning electrons to resultants...')
    resultants, dq = apportion_counts_to_resultants(
        counts.array, tij, inv_linearity=inv_linearity, crparam=crparam,
        persistence=persistence, tstart=tstart,
        rng=rng, seed=seed)

    # roman.addReciprocityFailure(resultants_object)

    add_ipc(resultants)

    if not isinstance(resultants, u.Quantity):
        resultants *= u.electron

    if gain is None:
        gain = parameters.reference_data['gain']
    if gain is not None and not isinstance(gain, u.Quantity):
        gain = gain * u.electron / u.DN
        log.warning('Making up units for gain.')

    resultants /= gain

    if read_noise is not None and not isinstance(read_noise, u.Quantity):
        read_noise = read_noise * u.DN
        log.warning('Making up units for read noise.')

    # resultants are now in counts.
    # read noise is in counts.
    log.info('Adding read noise...')
    resultants = add_read_noise_to_resultants(
        resultants, tij, rng=rng, seed=seed,
        read_noise=read_noise,
        pedestal_extra_noise=pedestal_extra_noise)

    # quantize
    resultants = np.round(resultants)

    # add pedestal
    resultants += parameters.pedestal

    if saturation is None:
        saturation = parameters.reference_data['saturation']

    # this maybe should be better applied at read time?
    # it's not actually clear to me what the right thing to do
    # is in detail.
    resultants = np.clip(resultants, 0 * u.DN, saturation)
    m = resultants >= saturation
    dq[m] |= parameters.dqbits['saturated']

    return resultants, dq
