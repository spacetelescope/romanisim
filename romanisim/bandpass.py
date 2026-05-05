"""Roman bandpass routines

The primary purpose of this module is to provide the number of electrons
per second expected for sources observed by Roman given a source with
the nominal flat AB spectrum of 3631 Jy.  The ultimate source of this
information is https://roman.gsfc.nasa.gov/science/WFI_technical.html .
"""

import warnings

warnings.warn(
    "romanisim.bandpass is deprecated and will be removed in a future version. "
    "Please use 'romanisim.models.bandpass' instead.",
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)

    from romanisim.models import bandpass

    warnings.warn(
        f"'romanisim.bandpass.{name}' is deprecated and will be removed in a future version."
        f"Please use 'romanisim.models.bandpass.{name}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(bandpass, name)