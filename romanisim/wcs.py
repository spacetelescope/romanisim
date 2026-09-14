"""Roman WCS interface for galsim.

galsim.roman has an implementation of Roman's WCS based on some SIP
coefficients for each SCA.  This could presumably be adequate, but here
we take the alternative approach of using the distortion functions
provided in CRDS.  These naturally are handled by the gWCS library,
but galsim only naturally supports astropy WCSes via the ~legacy
interface.  So this module primarily makes the wrapper that interfaces
gWCS and galsim.CelestialWCS together.

This presently gives rather different world coordinates given a specific
telescope boresight.  Partially this is not doing the same roll_ref
determination that galsim.roman does, which could be fixed.  But additionally
the center of the SCA looks to be in a different place relative to the
boresight for galsim.roman than for what I get from CRDS.  This bears
more investigation.
"""

import warnings

warnings.warn(
    "romanisim.wcs is deprecated and will be removed in a future version. "
    "Please use 'romanisim.models.wcs' instead.",
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
        
    from romanisim.models import wcs

    warnings.warn(
        f"'romanisim.wcs.{name}' is deprecated and will be removed in a future version. "
        f"Please use 'romanisim.models.wcs.{name}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(wcs, name)
