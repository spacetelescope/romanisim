"""Convert images into L1 images, with ramps.

We imagine starting with an image that gives the total number of counts from
all Poisson processes (at least: sky, sources, dark current).  We then need
to redistribute these counts over the resultants of an L1 image.

The easiest thing to do, and probably where I should start, is to sample the
image read-by-read with a binomial draw from the total counts weighted by the
chance that each count landed in this particular time window.  Then gather
those for each resultant and average, as done on the spacecraft.

It's tempting to go straight to making the appropriate resultants.  Following
Casertano (2022?), the variance in each resultant is:

.. math:: V = \sigma_{read}^2/N + f \\tau

where f is the count rate, N is the number of reads in the resultant, and :math:`\\tau`
is the 'variance-based resultant time'

.. math:: \\tau = 1/N^2 \sum_{reads} (2 (N - k) - 1) t_k

where the t_k is the time of the kth read in the resultant.

For uniformly spaced reads,

.. math:: \\tau = t_0 + d (N/3 + 1/6N - 1/2) \, ,

where t_0 is the time of the first read in the resultant and d is the spacing
of the reads.

So that gives the variance from Poisson counts in resultant.  But how should
we draw random numbers to get that variance and the right mean?  I can
separately control the mean and variance by scaling the Poisson distribution,
but I'm not sure that's doing the right thing with the higher order moments.

It probably isn't that expensive to just work out all of the reads,
individually, which will also allow more natural incorporation of
cosmic rays down the road.  So let's take that approach instead for
the moment.

How do we want to specify an L1 image?  An L1 image is defined by a
total count image and a list of lists :math:`t_{i, j}`, where
:math:`t_{i, j}` is the time at which the jth read in the ith
resultant is made.  We demand :math:`t_{i, j} > t_{k, l}`
whenever i > k or whenever i = k and j > l.

Things this doesn't allow neatly:

* jitter in telescope pointing: the rate image is the same for each read/resultant
* non-linearity?
* weird non-linear systematics in darks?

Possibly some systematics need to be applied to the individual reads, rather than to the final image.  e.g., clearly nonlinearity?  I need to think about when in the chain things like IPC, etc., come in.  But it still seems correct to first generate the total number of counts that an ideal detector would measure from sources, and then apply these effects read-by-read as we build up the resultants---i.e., I expect the current framework will be able to handle this without major issue.

This approach is not super fast.  For a high latitude set of resultants,
generating all of the random numbers to determine the apportionment takes
43 s on the machine I'm currently using; this will scale linearly with the
number of reads.  That's longer than the actual image production for the
dummy scene I'm using (though only ~2x longer).

I don't have a good way to speed this up.  Explicitly doing the Poisson
noise on each read from a rate image (rather than the apportionment of an
image that already has Poisson noise) is 2x slower---generating billions
of random numbers just takes a little while.

Plausibly I could figure out how to draw numbers directly from what a
resultant is rather than looking at each read individually.  That would
likely bring a ~10x speed-up.  The read noise there is easy.  The
poisson noise is a sum of scaled Poisson variables:

.. math:: \sum_{i=0, ..., N-1} (N-i) c_i \, ,

where :math:`c_i` is a Poisson-distributed variable.
The sum of Poisson-distributed variables is Poisson-distributed, but I wasn't
immediately able to find anything about the sum of scaled Poisson-distributed
variables.  The result is clearly not Poisson-distributed, but maybe there's
some special way to sample from that directly.

If we did sample from that directly, we'd still also need to get the sum
of the counts in the reads comprising the resultant.  So I think you'd need
a separate draw for that, conditional on the number you got for the resultant.
Or, reversing that, you might want to draw the total number of counts first,
e.g., via the binomial distribution, and then you'd want to draw a number
for what the average number of counts was among the reads comprising the
resultant, conditional on the total number of counts.  Then

.. math:: \sum_{i=0, ..., N-1} (N-i) c_i

is some kind of statistic of the multinomial distribution.  That sounds a
little more tractable?

.. math:: c_i \sim \mathrm{multinomial}(\mathrm{total}, [1/N, ..., 1/N])

We want to draw from :math:`\sum (N-i) c_i`.
I think the probabilities are always :math:`1/N`, with the possible small but
important
exception of 'skipped' or 'dropped' reads, in which case the first read would
be more like :math:`2/(N+1)` and all the others :math:`1/(N+1)`.  If the probabilities
were always 1/N, this looks vaguely like it could have a nice analytic
solution.  Otherwise, I don't immediately see a route forward.  So I am not
going to pursue this avenue further.
"""

import numpy as np
import asdf
import galsim
from galsim import roman
from . import parameters
from . import log
from . import util


def validate_times(tij):
    """Verify that a set of times tij for a valid resultant.

    Parameters
    ----------
    tij : list[list[float]]
        a list of list of times at which each read in a resultant is performed

    Returns
    -------
    bool
        True if the tij are ascending, otherwise False
    """
    times = [t for resultant in tij for t in resultant]
    return np.all(np.diff(times) > 0)


def tij_to_pij(tij):
    """Convert a set of times tij to corresponding probabilities for sampling.

    The probabilities are those needed for sampling from a binomial distribution
    for each read.  These are conceptually roughly delta_t / sum(delta_t), the
    fraction of time in each read over the total time, with one important
    difference.  Since we remove the counts that we've already allocated
    to reads from the number of counts remaining, the probabilities of
    subsequent reads get scaled up like so that each pij is
    delta_t / time_remaining.

    Parameters
    ----------
    tij : list[list[float]]
        list of list of readout times for each read entering a resultant

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
            pi.append((t - tlast) / tremaining)
            tremaining -= (t - tlast)
            tlast = t
        pij.append(pi)
    return pij


def apportion_counts_to_resultants(counts, tij):
    """Apportion counts to resultants given read times.

    This finds a statistically appropriate assignment of counts to each
    read composing a resultant, and averages the reads together to make
    the resultants.

    There's an alternative approach where you have a count rate image and
    need to do Poisson draws from it.  That's easier, and isn't this function.
    On some systems I've used Poisson draws have been slower than binomial
    draws, so it's not clear that approach offers any advantages, either---
    though I've had mixed experience there.

    We loop over the reads, each time sampling from the counts image according
    to the probability that a photon lands in that particular read.  This
    is just np.random.binomial(number of counts left, p/p_left)

    We then average the reads together to get a resulant.

    We accumulate:

    * a sum for the resultant, which is divided by the number of reads and
      returned in the resultants array
    * a sum for the total number of photons accumulated so far, so we know
      where to start the next resultant
    * the resultants so far

    Parameters
    ----------
    counts : np.ndarray[nx, ny] (int)
        The number of counts in each pixel from sources in the final image
        This final image should be a ~conceptual image of the scene observed
        by an idealized instrument seeing only backgrounds and sources and
        observing until the end of the last read; no instrumental effects are
        included beyond PSF & distortion.
    tij : list[list[float]]
        list of list of readout times for each read entering a resultant

    Returns
    -------
    np.ndarray[n_resultant, nx, ny]
        array of n_resultant images giving each resultant
    """

    if not np.all(counts == np.round(counts)):
        raise ValueError('apportion_counts_to_resultants expects the counts '
                         'to be integers!')

    pij = tij_to_pij(tij)
    resultants = np.zeros((len(tij),) + counts.shape, dtype='f4')
    counts_so_far = np.zeros(counts.shape, dtype='f4')
    resultant_counts = np.zeros(counts.shape, dtype='f4')

    for i, pi in enumerate(pij):
        resultant_counts[...] = 0
        for j, p in enumerate(pi):
            read = np.random.binomial((counts - counts_so_far).astype('i4'), p)
            counts_so_far += read
            resultant_counts += counts_so_far
        resultants[i, ...] = resultant_counts / len(pi)
    return resultants


def add_read_noise_to_resultants(resultants, tij, read_noise=None, rng=None,
                                 seed=None):
    """Adds read noise to resultants.

    The resultants get Gaussian read noise with sigma = sigma_read/sqrt(N).

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, nx, ny] (float)
        resultants array, giving each of n_resultant resultant images
    tij : list[list[float]]
        list of list of readout times for each read entering a resultant
    read_noise : float or np.ndarray[nx, ny] (float)
        read noise or read noise image for adding to resultants
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.

    Returns
    -------
    np.ndarray[n_resultant, nx, ny] (float)
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

    noise = np.empty(resultants.shape, dtype='f4')
    rng.generate(noise)
    if read_noise is None:
        read_noise = parameters.read_noise
    noise *= read_noise / np.array(
        [len(x)**0.5 for x in tij]).reshape(-1, 1, 1)
    resultants += noise
    return resultants


def make_asdf(resultants, filepath=None, metadata=None):
    """Package and optionally write out an L1 frame.

    This routine packages an L1 data file with the appropriate Roman data
    model.  It currently does not do anything with the necessary metadata,
    and leaves that information as filler values.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, nx, ny] (float)
        resultants array, giving each of n_resultant resultant images
    filepath : str
        if not None, path of asdf file to L1 image into

    Returns
    -------
    roman_datamodels.datamodels.ScienceRawModel
        L1 image
    """

    from roman_datamodels.testing.utils import mk_level1_science_raw
    out = mk_level1_science_raw(shape=(len(resultants), 4096, 4096))
    if metadata is not None:
        tmpmeta = util.flatten_dictionary(out['meta'])
        tmpmeta.update(util.flatten_dictionary(
            util.unflatten_dictionary(metadata)['roman']['meta']))
        out['meta'].update(util.unflatten_dictionary(tmpmeta))
    nborder = parameters.nborder
    out['data'][:, nborder:-nborder, nborder:-nborder] = resultants
    if filepath:
        af = asdf.AsdfFile()
        af.tree = {'roman': out}
        af.write_to(filepath)
    return out


def ma_table_to_tij(ma_table_number):
    """Get the times of each read going into resultants for a MA table.

    Currently only ma_table_number = 1 is supported, corresponding to a simple
    fiducial high latitude imaging MA table.

    This presently uses a hard-coded, somewhat inflexible MA table description
    in the parameters file.  But that seems like an okay option given that the
    current 'official' file is slated for redesign when the format is relaxed.

    Parameters
    ----------
    ma_table_number : int or list[list]
        if int, id of multiaccum table to use
        otherwise a list of (first_read, n_reads) tuples going into resultants.

    Returns
    -------
    list[list[float]]
        list of list of readout times for each read entering a resultant
    """
    if isinstance(ma_table_number, int):
        tab = parameters.ma_table[ma_table_number]
    else:
        tab = ma_table_number
    tij = [parameters.read_time * np.arange(f, f+n) for (f, n) in tab]
    return tij


def make_l1(counts, ma_table_number,
            read_noise=None, filepath=None, rng=None, seed=None):
    """Make an L1 image from a counts image.

    This apportions the total counts among the different resultants and adds
    some instrumental effects.  The current instrumental effects aren't quite
    right: nonlinearity and reciprocity failure are applied to the resultants
    rather than to the reads (which aren't really available to this function).

    Parameters
    ----------
    counts : galsim.Image
        total counts delivered to each pixel
    ma_table_number : int
        multi accum table number indicating how reads are apportioned among
        resultants
    filepath : str or None
        optional, path to which to write out L1 image
    read_noise : np.ndarray[nx, ny] (float) or float
        Read noise entering into each read
    rng : galsim.BaseDeviate
        Random number generator to use
    seed : int
        Seed for populating RNG.  Only used if rng is None.

    Returns
    -------
    np.ndarray[n_resultant, nx, ny]
        resultants image array including systematic effects
    """
    if rng is None and seed is None:
        seed = 45
        log.warning(
            'No RNG set, constructing a new default RNG from default seed.')
    if rng is None:
        rng = galsim.GaussianDeviate(seed)

    tij = ma_table_to_tij(ma_table_number)
    log.info('Apportioning counts to resultants...')
    resultants = apportion_counts_to_resultants(counts.array, tij)

    # we need to think harder around here about ndarrays vs. galsim
    # images.  All of the roman.function calls expect 2D galsim
    # images, but maybe wouldn't break with a 3D image.
    # lot's of im.array[:, :], but that seems to return the whole image?
    # from types import SimpleNamespace
    # resultants_object = SimpleNamespace()
    # resultants_object.array = resultants

    # these aren't quite right; we should be applying these to the reads
    # rather than to the resultants.
    # I can't decide how much to care---the effect is small.
    # The IPC computation is probably commutative, but the nonlinearity
    # corrections will not be.
    # roman.addReciprocityFailure(resultants_object)
    # roman.applyNonlinearity(resultants_object)
    # roman.applyIPC(resultants_object)
    # the roman systematic effects take a little more work
    # if we want to implement them directly on the individual resultants.
    # Those functions call namesakes in galsim.Image which the simple
    # namespace object above doesn't have, with specialized coefficients.
    # We could duplicate that, in principle.
    log.info('Adding read noise...')
    resultants = add_read_noise_to_resultants(resultants, tij, rng=rng,
                                              seed=seed, read_noise=read_noise)
    # presently in _electrons_ here; is read_noise in ADU or in electrons?!
    log.warning('We need to make sure we have the right units on the read '
                'noise.')

    resultants /= roman.gain
    resultants = np.round(resultants)
    return resultants
