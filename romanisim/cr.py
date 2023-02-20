import numpy as np
import scipy.interpolate as interpolate


def create_sampler(pdf, x):
    """A function for performing inverse transform sampling.

    Parameters
    ----------
    pdf : callable
        A function or empirical set of tabulated values which can
        be used to call or evaluate `x`.
    x : 1-d array of floats
        A grid of values where the pdf should be evaluated.

    Returns
    -------
    inverse_cdf : 1-d array of floats
        The cumulative distribution function which allows sampling
        from the `pdf` distribution within the bounds described
        by the grid `x`.
    """

    y = pdf(x)
    cdf_y = np.cumsum(y) - y[0]
    cdf_y /= cdf_y.max()
    inverse_cdf = interpolate.interp1d(cdf_y, x)
    return inverse_cdf


def moyal_distribution(x, location=120, scale=50):
    """Return unnormalized Moyal distribution, which approximates a
    Landau distribution and is used to describe the energy loss
    probability distribution of a charged particle through a detector.

    Parameters
    ----------
    x : 1-d array
        An array of dE/dx values (units: eV/micron) that forms the
        grid on which the Moyal distribution will be evaluated.
    location : float
        The peak location of the distribution, units of eV / micron.
    scale : float
        A width parameter for the distribution, units of eV / micron.
    Returns
    -------
    moyal : 1-d array of floats
        Moyal distribution (pdf) evaluated on `x` grid of points.
    """
    xs = (x - location) / scale
    moyal = np.exp(-(xs + np.exp(-xs)) / 2)
    return moyal


def power_law_distribution(x, slope=-4.33):
    """Return unnormalized power-law distribution parameterized by
    a log-log slope, used to describe the cosmic ray path lengths.

    Parameters
    ----------
    x : 1-d array of floats
        An array of cosmic ray path lengths (units: micron).
    slope : float
        The log-log slope of the distribution, default based on
        Miles et al. (2021).

    Returns
    -------
    power_law : 1-d array of floats
        Power-law distribution (pdf) evaluated on `x` grid of points.
    """
    power_law = np.power(x, slope)
    return power_law


def sample_cr_params(
    N_samples,
    N_i=4096,
    N_j=4096,
    min_dEdx=10,
    max_dEdx=10000,
    min_cr_len=10,
    max_cr_len=2000,
    grid_size=10000,
    rng=None,
    seed=48,
):
    """Generates cosmic ray parameters randomly sampled from distribution.
    One might re-implement this by reading in parameters from a reference
    file, or something similar.

    Parameters
    ----------
    N_samples : int
        Number of CRs to generate.
    N_i : int
        Number of pixels along i-axis of detector
    N_j : int
        Number of pixels along j-axis of detector
    min_dEdx : float
        Minimum value of CR energy loss (dE/dx), units of eV / micron.
    max_dEdx : float
        Maximum value of CR energy loss (dE/dx), units of eV / micron.
    min_cr_len : float
        Minimum length of cosmic ray trail, units of micron.
    max_cr_len : float
        Maximum length of cosmic ray trail, units of micron.
    grid_size : int
        Number of points on the cosmic ray length and energy loss grids.
        Increasing this parameter increases the level of sampling for
        the distributions.
    rng : np.random.Generator
        Random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    cr_x : float, between 0 and N_x-1
        x pixel coordinate of cosmic ray, units of pixels.
    cr_y : float between 0 and N_y-1
        y pixel coordinate of cosmic ray, units of pixels.
    cr_phi : float between 0 and 2pi
        Direction of cosmic ray, units of radians.
    cr_length : float
        Cosmic ray length, units of micron.
    cr_dEdx : float
        Cosmic ray energy loss, units of eV / micron.
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    # sample CR positions [pix]
    cr_i, cr_j = (
        rng.random(size=(N_samples, 2)) * (N_i, N_j) - 0.5
    ).transpose()

    # sample CR direction [radian]
    cr_phi = rng.random(N_samples) * 2 * np.pi

    # sample path lengths [micron] 
    len_grid = np.linspace(min_cr_len, max_cr_len, grid_size)
    inv_cdf_len = create_sampler(power_law_distribution, len_grid)
    cr_length = inv_cdf_len(rng.random(N_samples))

    # sample energy losses [eV/micron]
    dEdx_grid = np.linspace(min_dEdx, max_dEdx, grid_size)
    inv_cdf_dEdx = create_sampler(moyal_distribution, dEdx_grid)
    cr_dEdx = inv_cdf_dEdx(rng.random(N_samples))

    return cr_i, cr_j, cr_phi, cr_length, cr_dEdx


def traverse(trail_start, trail_end, N_i=4096, N_j=4096):
    """Given a starting and ending pixel, returns a list of pixel
    coordinates (ii, jj) and their traversed path lengths. Note that
    the centers of pixels are treated as integers, while the borders
    are treated as half-integers.

    Parameters
    ----------
    trail_start : (float, float)
        The starting coordinates in (i, j) of the cosmic ray trail,
        in units of pix.
    trail_end : (float, float)
        The ending coordinates in (i, j) of the cosmic ray trail, in
        units of pix.
    N_i : int 
        Number of pixels along i-axis of detector
    N_j : int 
        Number of pixels along j-axis of detector
    eps : float 
        Tiny value used for stable numerical rounding.

    Returns
    -------
    ii : np.ndarray[int]
        i-axis positions of traversed trail, in units of pix.
    jj : np.ndarray[int]
        j-axis positions of traversed trail, in units of pix.
    lengths : np.ndarray[float]
        Chord lengths for each traversed pixel, in units of pix.
    """

    # increase in i-direction
    if trail_start[0] < trail_end[0]:
        i0, j0 = trail_start
        i1, j1 = trail_end
    else:
        i1, j1 = trail_start
        i0, j0 = trail_end
    
    di = i1 - i0
    dj = j1 - j0
    slope = dj / di
    sign = np.sign(slope)

    # horizontal border crossings at j = integer + 1/2 
    if dj != 0:
        j_horiz = np.arange(np.round(j0), np.round(j1), sign) + 0.5 * sign
        i_horiz = i0 + (di / dj) * (j_horiz - j0)
        cross_horiz = np.transpose([i_horiz, j_horiz])
    else:
        cross_horiz = np.array([[]])

    # vertical border crossings at i = integer + 1/2
    if di != 0:
        i_vert = np.arange(np.round(i0), np.round(i1), 1) + 0.5
        j_vert = j0 + (dj / di) * (i_vert - i0)
        cross_vert = np.transpose([i_vert, j_vert])
    else:
        cross_vert = np.array([[]])
    
    # compute center of traversed pixel before each crossing
    # note `eps` here covers weird rounnding issues when the corner is intersected
    ii_horiz, jj_horiz = np.round(
        cross_horiz - np.array([eps, np.sign(dj)*0.5])
    ).astype(int).T
    ii_vert, jj_vert = np.round(
        cross_vert - np.array([0.5, np.sign(dj) * eps])
    ).astype(int).T

    # combine crossings and pixel centers
    crossings = np.vstack((cross_horiz, cross_vert))
    ii = np.concatenate((ii_horiz, ii_vert))
    jj = np.concatenate((jj_horiz, jj_vert))

    # sort by i axis
    sorted_by_i = np.argsort(crossings[:, 0])
    crossings = crossings[sorted_by_i]
    ii = ii[sorted_by_i]
    jj = jj[sorted_by_i]

    # add final pixel center
    ii = np.concatenate((ii, [np.round(i1).astype(int)]))
    jj = np.concatenate((jj, [np.round(j1).astype(int)]))
    
    # if no crossings, then it's just the total Euclidean distance
    if len(crossings) == 0:
        lengths = np.linalg.norm([di, dj], keepdims=1)
    # otherwise, compute starting, crossing, and ending distances
    else:
        first_length = np.linalg.norm(crossings[0] - np.array([i0, j0]), keepdims=1)
        middle_lengths = np.linalg.norm(np.diff(crossings, axis=0), axis=1)
        last_length = np.linalg.norm(np.array([i1, j1]) - crossings[-1], keepdims=1)
        lengths = np.concatenate([first_length, middle_lengths, last_length])

        # remove 0-length trails
        positive_lengths = lengths > 0
        lengths = lengths[positive_lengths]
        ii = ii[positive_lengths]
        jj = jj[positive_lengths]

    # remove any pixels that go off the detector
    inside_detector = (ii > -0.5) & (ii < (N_i - 0.5)) & (jj > +0.5) & (jj < (N_j + 0.5))
    ii = ii[inside_detector]
    jj = jj[inside_detector]
    lengths = lengths[inside_detector]
    
    return ii, jj, lengths


def simulate_crs(
    image, 
    time, 
    flux=8, 
    area=16.8, 
    conversion_factor=0.5, 
    pixel_size=10,
    pixel_depth=5, 
    rng=None, 
    seed=47
):
    """Adds CRs to an existing image.

    Parameters
    ----------
    image : 2-d array of floats
        The detector image with values in units of counts.
    time : float
        The exposure time, units of s.
    flux : float
        Cosmic ray flux, units of cm^-2 s^-1. Default value of 8
        is equal to the value assumed by the JWST ETC.
    area : float
        The area of the WFI detector, units of cm^2.
    conversion_factor : float 
        The convert from eV to counts, assumed to be the bandgap energy,
        in units of eV / counts.
    pixel_size : float
        The size of an individual pixel in the detector, units of micron.
    pixel_depth : float
        The depth of an individual pixel in the detector, units of micron.
    rng : np.random.Generator
        Random number generator to use
    seed : int
        seed to use for random number generator

    Returns
    -------
    image : 2-d array of floats
        The detector image, in units of counts, updated to include
        all of the generated cosmic ray hits.
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    N_i, N_j = image.shape
    N_samples = rng.poisson(flux * area * time)
    cr_i0, cr_j0, cr_angle, cr_length, cr_dEdx = sample_cr_params(
        N_samples, N_i=N_i, N_j=N_j, rng=rng)

    cr_length = cr_length / pixel_size
    cr_i1 = (cr_i0 + cr_length * np.cos(cr_angle)).clip(-0.5, N_i + 0.5)
    cr_j1 = (cr_j0 + cr_length * np.sin(cr_angle)).clip(-0.5, N_j + 0.5)

    # go from eV/micron -> counts/pixel
    cr_counts_per_pix = cr_dEdx * pixel_size / conversion_factor
    
    for i0, j0, i1, j1, counts_per_pix in zip(cr_i0, cr_j0, cr_i1, cr_j1, cr_counts_per_pix):
        ii, jj, length_2d = traverse([i0, j0], [i1, j1], N_i=N_i, N_j=N_j)
        length_3d = ((pixel_depth / pixel_size) ** 2 + length_2d ** 2) ** 0.5
        image[ii, jj] += rng.poisson(counts_per_pix * length_3d)

    return image


if __name__ == "__main__":
    # initialize a detector
    image = np.zeros((4096, 4096), dtype=float)

    flux = 8  # events/cm^2/s
    area = 0.168  # cm^2
    t_exp = 3.04  # s

    # simulate 500 resultant frames
    for _ in range(500):
        image = simulate_crs(image, flux, area, t_exp)
