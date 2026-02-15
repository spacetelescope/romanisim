import os

import asdf
import crds
import galsim
import roman_datamodels

from astropy import units as u
from astropy.io import ascii

from .gain import gain
from .parameters import (
    default_parameters_dictionary,
    nborder,
    roman_tech_repo_path,
)

__all__ = ["DarkCurrent"]

# Default dark current
dark_current = 0.015  # e-/pix/s

# Update dark current value with one from roman-technical-information
# Columns in the summary file: ['SCU', 'SCA', 'Dark Current - Median', 'Dark Current - Mean', 'Percentage Passing Requirement']
# The 18th (counting from 0) row: All detectors (MAP)
dark_current_summary = os.path.join(
    roman_tech_repo_path,
    "data",
    "WideFieldInstrument",
    "FPSPerformance",
    "WFI_Dark_current_summary.ecsv",
)
try:
    data = ascii.read(dark_current_summary)
    dark_current = data[18]["Dark Current - Median"]
except RuntimeError as e:
    print(
        f" {e} Failed to fetch WFI_Dark_current_summary.ecsv, use default value for dark_current"
    )


class DarkCurrent(object):
    """
    Add Roman/WFI dark current signal and Poisson noise to an image.

    The dark current can be represented either as:
      - a scalar rate (e-/pix/s), or
      - a 2D per-pixel array of rates (e-/pix/s) loaded from CRDS.

    Parameters
    ----------
    usecrds : bool, optional
        If True, query CRDS for Roman "dark" and "gain" reference files and
        compute a per-pixel dark rate in electrons/second. If False, use the
        module-level default (possibly updated from roman-technical-information).
    getdq : bool, optional
        If True and ``usecrds=True``, also read the ``roman/dq`` array from the
        CRDS dark reference and store it as ``self.dq`` (cropped by ``nborder``).
        Ignored when ``usecrds=False``.
    metadata : dict or None, optional
        Metadata overrides to apply before CRDS lookup. If provided, values
        are merged into the model metadata tree.
    rng : galsim.BaseDeviate or compatible, optional
        Random number generator to seed the Poisson noise. If provided, it is
        wrapped as `galsim.BaseDeviate(rng)`.
    seed : int or None, optional
        Seed used to initialize the RNG when `rng` is not provided.

    Attributes
    ----------
    dark_rate : float or numpy.ndarray
        Dark current rate in electrons per pixel per second. Scalar if CRDS is
        not used; 2D array if CRDS is used.
    gain : float or numpy.ndarray
        Gain used when converting CRDS dark model into electrons/sec. Defaults
        to `.gain.gain` unless overwritten by CRDS gain reference.
    dq : numpy.ndarray
        Only present if ``usecrds=True`` and ``getdq=True``. The DQ array read
        from the CRDS dark reference, cropped by ``nborder``.
    rng : galsim.BaseDeviate
        RNG used for Poisson noise.
    """

    def __init__(
        self, usecrds=False, getdq=False, metadata=None, rng=None, seed=None
    ):
        self.dark_rate = dark_current
        self.gain = gain
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata, getdq=getdq)

        if rng is None and seed is None:
            self.seed = 45
        if rng is None:
            self.rng = galsim.BaseDeviate(seed)
        else:
            self.rng = galsim.BaseDeviate(rng)

    def _get_crds_model(self, getdq=False, metadata=None):
        """
        Load CRDS dark-current and gain reference files and compute dark rate.

        This method:
          1) Creates a fake `roman_datamodels` ImageModel and populates its
             `meta` using `default_parameters_dictionary`.
          2) Applies any user-provided `metadata` overrides.
          3) Uses CRDS to locate "dark" and "gain" reference files.
          4) Reads:
             - `roman/dark_slope` from the dark reference, cropped by `nborder`,
             - `roman/data` from the gain reference, cropped by `nborder`.
          5) Converts the dark slope into electrons/second by multiplying by
             the gain and stripping any astropy units.

        Parameters
        ----------
        getdq : bool, optional
            If True, also read ``roman/dq`` from the dark reference and store it
            as ``self.dq``.
        metadata : dict or None, optional
            Metadata overrides applied to the fake model's `meta` before CRDS
            reference selection.

        Side Effects
        ------------
        Sets `self.dark_rate` to a 2D array (e-/s) and `self.gain` to a 2D array.
        """
        image_mod = roman_datamodels.datamodels.ImageModel.create_fake_data()
        meta = image_mod.meta
        meta["wcs"] = None
        for key in default_parameters_dictionary.keys():
            meta[key].update(default_parameters_dictionary[key])

        if metadata:
            for key in metadata.keys():
                meta[key].update(metadata[key])

        ref_file = crds.getreferences(
            image_mod.get_crds_parameters(),
            reftypes=["dark", "gain"],
            observatory="roman",
        )

        print(ref_file)

        with asdf.open(ref_file["dark"]) as f:
            self.dark_rate = f["roman"]["dark_slope"][
                nborder:-nborder, nborder:-nborder
            ].copy()
            if getdq:
                self.dq = f["roman"]["dq"][
                    nborder:-nborder, nborder:-nborder
                ].copy()
        with asdf.open(ref_file["gain"]) as f:
            self.gain = f["roman"]["data"][
                nborder:-nborder, nborder:-nborder
            ].copy()
        # self.dark_rate * u.DN / u.s
        self.dark_rate *= self.gain
        # if isinstance(self.dark_rate, u.Quantity):
        #     self.dark_rate = self.dark_rate.to(u.electron / u.s).value

    def apply(self, img, exptime):
        """
        Add dark current signal and Poisson noise to an image (in place).

        The method creates a zero-valued working image with the same shape as
        `img`, adds the expected dark current signal (`dark_rate * exptime`),
        then applies Poisson noise using `galsim.PoissonNoise(self.rng)`.
        Finally, it adds the noisy dark-current realization back into the input.

        Parameters
        ----------
        img : galsim.Image or array-like (2D)
            Input image to be modified in place. If a `galsim.Image`, the result
            is accumulated back into that object. If an array-like, it is
            converted to `galsim.Image` internally and the result is added back
            to the original array via `img += workim.array`.
        exptime : float
            Exposure time in seconds.

        Notes
        -----
        - Units: `dark_rate` is treated as electrons/pixel/second, so the added
          signal is in electrons/pixel.
        - This function modifies `img` in place and returns None.

        Returns
        -------
        None
            The input `img` is modified in place.
        """
        if isinstance(img, galsim.Image):
            workim = img * 0
        else:
            workim = galsim.Image(img) * 0
        workim += self.dark_rate * exptime
        workim.addNoise(galsim.PoissonNoise(self.rng))
        if isinstance(img, galsim.Image):
            img += workim
        else:
            img += workim.array
