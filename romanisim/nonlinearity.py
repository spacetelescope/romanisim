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

import warnings

warnings.warn(
    "romanisim.nonlinearity is deprecated and will be removed in a future version. "
    "Please use 'romanisim.models.nonlinearity' instead.",
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)

    if name == 'NL':
        raise ValueError("romanisim.nonlinearity.NL' is deprecated. Please use romanisim.models.nonlinearity.Nonlinearity instead")
        
    from romanisim.models import nonlinearity

    warnings.warn(
        f"'romanisim.nonlinearity.{name}' is deprecated and will be removed in a future version. "
        f"Please use 'romanisim.models.nonlinearity.{name}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(nonlinearity, name)
