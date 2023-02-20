"""Persistence module.

This module implements a persistence simulation following Sanchez+2023.
"""

from . import parameters
import numpy as np


class Persistence:
    """Track persistence information.

    There are two important sets of things to keep track of with persistence:
    - how pixels respond to persistence
    - what pixels have experienced large fluxes in the past and may be affected
      by persistence.
    This class tracks both of those quantities.  The first category is expected
    to be largely constant with time and basically only a function of the specific
    device doing the imaging.  The second category changes in each exposure as
    new bright stars are observed.
    """
    def __init__(self, x=None, t=None, index=None, A=None, x0=None, dx=None,
                 alpha=None, gamma=None):
        """Construct a new Persistence instance.

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
        if x is None:
            x = np.zeros(0, dtype='f4')
        if t is None:
            t = np.zeros(0, dtype='f8')
        if index is None:
            index = np.zeros(0, dtype='i4')
        if A is None:
            A = parameters.persistence['A']
        if x0 is None:
            x0 = parameters.persistence['x0']
        if dx is None:
            dx = parameters.persistence['dx']
        if alpha is None:
            alpha = parameters.persistence['alpha']
        if gamma is None:
            gamma = parameters.persistence['gamma']
        if ((np.array(x).shape != np.array(t).shape)
            or (np.array(x).shape != np.array(index).shape)):
            raise ValueError('x, t, index must have identical shapes!')
        self.x = x
        self.t = t
        self.index = index
        self.A = A
        self.x0 = x0
        self.dx = dx
        self.alpha = alpha
        self.gamma = gamma

    def add_to_read(self, image, tnow, rng=None, seed=50):
        """Add persistence signature to image.
        
        Parameters
        ----------
        image : np.ndarray[float], shape: (npix_x, npix_y)
            Image to which to add persistence (electrons)
        tnow : float
            Current time (MJD)
        rng : np.random.Generator
            Random number generator
        seed : int
            Seed to use if instantiating new random number generator.
        """

        if rng is None:
            rng = np.random.default_rng(seed)

        nx, ny = image.shape
        image = image.reshape(-1)
        persistence = self.current(tnow)
        persistence = rng.poisson(persistence * parameters.read_time)
        np.add.at(image, self.index, persistence)

    def current(self, tnow):
        """Evaluate current in electron / s from past persistence artifacts
        at time tnow.

        Parameters
        ----------
        tnow : float
            Current time (MJD)

        Returns
        -------
        Current in electron / s in pixels due to past persistence events.
        """
        return fermi(self.x, (tnow - self.t) * 60 * 60 * 24,
                     self.A, self.x0, self.dx, self.alpha, self.gamma)
    

    def update(self, image, tnow):
        """Update stored fluence values of events worth tracking for future 
        persistence.

        New persistence-affected pixels are added and old ones removed
        according to whether the predicted persistence rate is larger than 
        parameters.persistence['ignorerate'].

        Parameters
        ----------
        image : np.ndarray[float]
            Image of total electrons accumulated in exposure
        tnow : float
            MJD of current observation
        """
        newrate = fermi(image.reshape(-1), 30, self.A, self.x0, self.dx,
                        self.alpha, self.gamma)
        idx = np.flatnonzero(
            newrate > parameters.persistence['ignorerate'])
        newx = image.reshape(-1)[idx]
        newt = tnow * np.ones(len(idx))
        newidx = idx

        oldrate = self.current(tnow + 30 / 60 / 60 / 24)
        oldidx = np.flatnonzero(oldrate > parameters.persistence['ignorerate'])

        self.x = np.concatenate([newx, self.x[oldidx]])
        self.t = np.concatenate([newt, self.t[oldidx]])
        self.index = np.concatenate([newidx, self.index[oldidx]])


def fermi(x, dt, A, x0, dx, alpha, gamma):
    """
    The Fermi model for persistence:
    A * (x/x0)**alpha * (t/1000.)**(-gamma) / (exp(-(x-x0)(/dx) + 1)
    For influence level below the half well, the persistence is linear in x.
    
    Parameters
    ----------
    x : np.ndarray[float]
        Fluence level (electrons)
    dt : np.ndarray[float]
        Time since exposure (s)
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

    Returns:
    --------
    The persistence signal at the current time for the persistence-affected pixels
    described by persistence.
    """
    scalar = np.isscalar(x)
    x = np.atleast_1d(x)
    hw = parameters.persistence['half_well']
    m = x < hw
    out = (A / (1 + np.exp(-(x - x0) / dx))
           * (x / x0)**alpha * (dt / 1000)**(-gamma))
    outhf = (A / (1 + np.exp(-(hw - x0) / dx))
             * (hw / x0)**alpha * (dt / 1000)**(-gamma))
    out[m] = (outhf * x / hw)[m]
    if scalar:
        out = out[0]
    return out

