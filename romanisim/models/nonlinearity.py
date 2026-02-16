import crds
import galsim
import numpy as np

from astropy import units as u
from roman_datamodels import datamodels

from .gain import gain
from .parameters import default_parameters_dictionary, dqbits, nborder

__all__ = ["NLfunc", "Nonlinearity"]

# Default nonlinearity beta value
nonlinearity_beta = -6.0e-7


# def print_ram_usage(message=""):
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()

#     print(f"{message}, RSS (Resident Set Size): {mem_info.rss / 1024 / 1024:.2f} MB")
#     print(f"{message}, VMS (Virtual Memory Size): {mem_info.vms / 1024 / 1024:.2f} MB")


def NLfunc(x):
    return x + nonlinearity_beta * (x**2)


class Nonlinearity(object):
    """
    Apply Roman/WFI classical nonlinearity correction using polynomial coefficients.

    The correction is represented as a per-pixel polynomial evaluated with
    `numpy.polyval`. By default a simple 2-term polynomial is used:

        y = 1 * x + beta * x^2

    where ``beta`` is the module-level ``nonlinearity_beta``. When
    ``usecrds=True``, per-pixel coefficients are loaded from the Roman CRDS
    *inverse linearity* reference (reftype ``inverselinearity``). A gain map
    can also be loaded from CRDS (reftype ``gain``), allowing the correction
    to be applied to images stored in electrons.

    Parameters
    ----------
    usecrds : bool, optional
        If True, query CRDS and load per-pixel polynomial coefficients from the
        ``inverselinearity`` reference and a gain map from the ``gain`` reference.
        If False, use the module defaults.
    getdq : bool, optional
        If True and ``usecrds=True``, also read the DQ array from the CRDS
        inverselinearity file and store it as ``self.dq`` (cropped by ``nborder``).
    metadata : dict or None, optional
        Metadata overrides to apply before CRDS lookup. If provided, values
        are merged into the model metadata tree.

    Attributes
    ----------
    coeffs : numpy.ndarray
        Nonlinearity polynomial coefficients with shape ``(ncoeff, ny, nx)``.
        Coefficients are stored in *increasing* power order
        (constant term first), consistent with the source CRDS file and the
        usage in `_evaluate_nl_polynomial` (which reverses by default for
        `numpy.polyval`).
    gain : float or numpy.ndarray
        Gain used to convert between electrons and DN when `electrons=True` in
        `apply()`. Defaults to `.gain.gain` unless replaced by CRDS gain map.
    dq : numpy.ndarray or None
        Only set when ``usecrds=True`` (and read from CRDS). If `getdq=False`,
        may be `None`. Note: current code does not always propagate/update DQ
        flags unless `getdq=True` is passed into `_repair_coefficients`.
    """

    def __init__(
        self,
        usecrds=False,
        reftype="inverselinearity",
        getdq=False,
        integralnonlinearity=False,
        metadata=None,
        image_mod=None,
        reffiles=None,
        saturation=None,
    ):
        self.gain = gain
        self.usecrds = usecrds
        self.metadata = metadata
        self.coeffs = np.array([1.0, nonlinearity_beta])
        self.saturation = saturation
        self.reftype = reftype
        self.integralnonlinearity = integralnonlinearity
        self.inl_corrs = None
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata, getdq=getdq, image_mod=image_mod, reffiles=reffiles)

    def _get_crds_model(self, getdq=False, metadata=None, image_mod=None, reffiles=None):
        # Inverse linearity reference files are used to apply the
        # effect of classical non-linearity when constructing
        # L1 files, and linearity reference files are used to
        # remove it when constructing L2 files.
        
        if self.integralnonlinearity:
            reftypes = [self.reftype, "gain", "integralnonlinearity"]
        else:
            reftypes = [self.reftype, "gain"]
        
        if image_mod is not None:
            ref_file = crds.getreferences(
                image_mod.get_crds_parameters(),
                reftypes=reftypes,
                observatory="roman",
            )
        elif reffiles is not None:
            ref_file = reffiles
        else:
            image_mod = datamodels.ImageModel.create_fake_data()
            meta = image_mod.meta
            meta["wcs"] = None
            for key in default_parameters_dictionary.keys():
                meta[key].update(default_parameters_dictionary[key])
            if metadata:
                for key in metadata.keys():
                    meta[key].update(metadata[key])
            ref_file = crds.getreferences(
                image_mod.get_crds_parameters(),
                reftypes=reftypes,
                observatory="roman",
            )
        
        if isinstance(ref_file['gain'], str):
            model = datamodels.open(ref_file['gain'])
            self.gain = model.data[nborder:-nborder, nborder:-nborder].copy()
        
        if isinstance(ref_file[self.reftype], str):
            nl_model = datamodels.open(ref_file[self.reftype])
            if getdq:
                self.dq = nl_model.dq[nborder:-nborder, nborder:-nborder].copy()
            else:
                self.dq = None
            self.coeffs = self._repair_coefficients(
                coeffs=nl_model.coeffs[
                    :, nborder:-nborder, nborder:-nborder
                ].copy(),
                dq=self.dq,
            )

        if self.integralnonlinearity and isinstance(ref_file["integralnonlinearity"], str):
            inl_model = datamodels.open(ref_file["integralnonlinearity"])
            # with asdf.open(ref_file["integralnonlinearity"]) as f:
            channel_width = 128
            ncols = self.coeffs.shape[2]
            self.inl_lookup = inl_model.value.copy()
            self.inl_corrs = {}
            sign = -1 if self.reftype == "inverselinearity" else 1
            for start_col in range(0, ncols, channel_width):
                channel_num = start_col // channel_width + 1
                attr_name = f"science_channel_{channel_num:02d}"
                self.inl_corrs[channel_num] = (
                    sign
                    * getattr(inl_model.inl_table, attr_name).correction.copy()
                )

    def _repair_coefficients(self, coeffs, dq, getdq=False):
        """Fix cases of zeros and NaNs in non-linearity coefficients.

        This function replaces suspicious-looking non-linearity coefficients
        with identity transformation coefficients from a non-linearity
        perspective; all coefficients are zero except for the linear term,
        which is set to 1.

        This function doesn't try to make sure that the derivative of the
        correction is greater than 1, which we would expect for a non-linearity
        correction.

        Parameters
        ----------
        coeffs : np.ndarray[ncoeff, ny, nx] (float)
            Nonlinearity coefficients, starting with the constant term and
            increasing in power.
        getdq : bool, optional
            If True, additionally OR in a DQ bit (``dqbits["no_lin_corr"]``) for
            pixels that were repaired, and store the result in ``self.dq``.

        dq : np.ndarray[n_resultant, ny, nx]
            Data Quality array

        Returns
        -------
        coeffs : np.ndarray[ncoeff, ny, nx] (float)
            "repaired" coefficients with NaNs and weird coefficients replaced
            with linear values with slopes of unity.
        """
        res = coeffs.copy()

        if dq is None:
            dq = np.zeros(coeffs.shape[1:], dtype=np.uint32)

        nocorrection = np.zeros(coeffs.shape[0], dtype=coeffs.dtype)
        nocorrection[1] = 1.0  # "no correction" is just normal linearity.
        # For NaN, all zero, or flagged pixels, reset to no correction.
        m = (
            np.any(~np.isfinite(coeffs), axis=0)
            | np.all(coeffs == 0, axis=0)
            | (dq != 0)
        )
        res[:, m] = nocorrection[:, None]

        # [TODO] deal with dq
        if getdq:
            lin_dq_array = np.zeros(coeffs.shape[1:], dtype=np.uint32)
            lin_dq_array[m] = dqbits["no_lin_corr"]
            self.dq = np.bitwise_or(dq, lin_dq_array)
        return res

    def _evaluate_nl_polynomial(self, counts, coeffs, reversed=False):
        """Correct the observed DN for non-linearity.

        As electrons accumulate, they make it harder for the device to count
        future electrons due to classical non-linearity.  This function
        converts observed DN to what would have been seen absent
        non-linearity, using the provided non-linearity coefficients.

        Parameters
        ----------
        counts : np.ndarray[ny, nx] (float)
            Number of DN already in pixel
        coeffs : np.ndarray[ncoeff, ny, nx] (float)
            Coefficients of the non-linearity correction polynomials
        reversed : bool
            If True, the coefficients are in reversed order, which is the
            order that np.polyval wants them.  One can maybe save a little
            time reversing them once ahead of time.

        Returns
        -------
        corrected : np.ndarray[nx, ny] (float)
            The corrected number of DN
        """
        if reversed:
            cc = coeffs
        else:
            cc = coeffs[::-1, ...]

        if isinstance(counts, u.Quantity):
            unit = counts.unit
            counts = counts.value
        else:
            unit = None

        res = np.polyval(cc, counts)

        if unit is not None:
            res = res * unit

        return res

    def apply(self, img, electrons=False, reversed=False):
        """Compute the correction of DN to linearized DN.

        Alternatively, when electrons = True, rescale these to DN,
        correct the DN, and scale them back to electrons using
        the gain.

        Parameters
        ----------
        img : numpy.ndarray or galsim.Image
            The observed img

        electrons : bool
            Set to True for 'img' being in electrons, with coefficients
            designed for DN. Accordingly, the gain needs to be removed and
            reapplied.

        reversed : bool
            If True, the coefficients are in reversed order, which is the
            order that np.polyval wants them.  One can maybe save a little
            time reversing them once ahead of time.
        """

        if isinstance(img, galsim.Image):
            img_arr = img.array
        else:
            img_arr = img

        if electrons:
            img_arr = img_arr / self.gain

        if self.saturation is not None:
            img_arr = np.clip(img_arr, -1000, self.saturation)

        corrected = self._evaluate_nl_polynomial(
            img_arr, self.coeffs, reversed
        )

        if self.inl_corrs is not None and img_arr.ndim >= 2:
            corrected = corrected + self.inl_correction(img_arr)

        if electrons:
            corrected = corrected * self.gain

        if isinstance(img, galsim.Image):
            img.array = corrected
        else:
            img = corrected
        return img

    def inl_correction(self, counts):
        """Compute the integral nonlinearity correction.

        Parameters
        ----------
        counts : np.ndarray
            The counts in DN to compute the correction for.

        Returns
        -------
        correction : np.ndarray
            The INL correction to be added to the counts.
        """
        channel_width = 128
        ncols = counts.shape[-1]
        correction = np.zeros_like(counts)
        for start_col in range(0, ncols, channel_width):
            channel_num = start_col // channel_width + 1
            channel_corr = self.inl_corrs[channel_num]
            channel_data = counts[..., start_col : start_col + channel_width]
            correction[..., start_col : start_col + channel_width] = np.interp(
                channel_data, self.inl_lookup, channel_corr
            )
        return correction
