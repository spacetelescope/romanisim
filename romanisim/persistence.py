"""Persistence module.

This module implements a persistence simulation following Sanchez+2023.
"""

import dataclasses
from . import parameters
import numpy as np


@dataclasses.dataclass
class Persistence:
    """Dataclass for holding persistence parameters.

    Parameters
    ----------
    x : np.ndarray[float]
        Fluence level (electrons)
    t : np.ndarray[float]
        Time since exposure (s)
    index : np.ndarray[integer]
        Indices of persistence-affected pixels (1D into raveled array)
    A : float
        Amplitude parameter of persistence (electrons)
    x0 : float
        Pivot fluence (electrons)
    dx : float
        dx parameter (electrons)
    alpha : float
        Power law index scaling with fluence
    gamma : float
        Power law index scaling with time
    """
    x: np.ndarray
    t: np.ndarray
    index: np.ndarray
    A = parameters.persistence['A']: float
    x0 = parameters.persistence['x0']: float
    dx = parameters.persistence['dx']: float
    alpha = parameters.persistence['alpha']: float
    gamma = parameters.persistence['gamma']: float


def fermi(persistence, t):
    """
    The Fermi model for persistence:
    A * (x/x0)**alpha * (t/1000.)**(-gamma) / (exp(-(x-x0)/dx) + 1)
    For influence level below the half well, the persistence is linear in x.
    
    Parameters
    ----------
    persistence : Persistence
        Persistence object containing information about persistence-affected 
        pixels.
    t : float
        Current time (MJD)

    Returns:
    --------
    The persistence signal at the current time for the persistence-affected pixels
    described by persistence.
    """

    return (
        persistence.A
        / (1 + np.exp(-(persistence.x - persistence.x0) / persistence.dx))
        * (persistence.x / persistence.x0)**persistence.alpha
        * ((t - persistence.t) * 24 * 60 * 60 / 1000)**persistence.gamma)


def add_persistence_to_read(image, tnow, persistence, rng=None, seed=None):
    """
    Add persistence signature to image.
    
    Parameters
    ----------
    image : np.ndarray[float], shape: (npix_x, npix_y)
        Image to which to add persistence (electrons)
    tnow : float
        Current time (MJD)
    persistence : Persistence
        Persistence instance describing persistence-affected pixels.
    rng : np.random.Generator
        Random number generator
    seed : int
        Seed to use if instantiating new random number generator.
    """

    nx, ny = image.shape
    image = image.reshape(-1)
    persistence = fermi(persistence, tnow) * parameters.read_time
    persistence = rng.poisson(persistence)
    np.add.at(img, index, persistence)


def update_persistence(image, tnow, persistence=None):
    """Update stored fluence values of events worth tracking for future 
    persistence.

    New persistence-affected pixels are added and old ones removed according
    to whether the predicted persistence rate is larger than 
    parameters.persistence['ignorerate'].

    Parameters
    ----------
    image : np.ndarray[float]
        Image of total electrons accumulated in exposure
    tnow : float
        MJD of current observation
    persistence : Persistence
        Persistence instance describing persistence-affected pixels.

    Returns
    -------
    Persistence
        Persistence instance updated with new persistence-affected pixels added
        and sufficiently old no-longer affected pixels removed.
    """
    newpersistence = Persistence(image.ravel(), np.array(tnow),
                                 np.arange(len(image.reshape(-1)), dtype='i4'))
    newrate = fermi(newpersistence, tnow + 60 / (24 * 60 * 60))
    idx = np.flatnonzero(
        newrate > parameters.persistence['ignorerate'])
    newpersistence.x = newpersistence.x[idx]
    newpersistence.t = np.ones(len(idx)) * tnow
    newpersistence.index = newpersistence.index[idx]

    if persistence is not None:
        oldrate = fermi(persistence, tnow)
        old_idx = np.flatnonzero(
            oldrate > parameters.persistence['ignorerate'])
        for field in ['x', 't', 'index']:
            old = getattr(persistence, field)[old_idx]
            setattr(newpersistence, field,
                    np.concatenate([old, getattr(newpersistence, field)]))
        for field in ['A', 'x0', 'dx', 'alpha', 'gamma']:
            setattr(newpersistence, field, getattr(oldpersistence, field))
    return newpersistence
