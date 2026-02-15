import asdf
import crds
import galsim
import roman_datamodels

from .parameters import default_parameters_dictionary, nborder

__all__ = ["Gain"]

# Default gain value
gain = 2.0


class Gain(object):
    """Detector gain model.

    Parameters
    ----------
    usecrds : bool, optional
        If ``True``, load the gain reference data from CRDS using
        :mod:`roman_datamodels` to construct the required CRDS parameter set.
        If ``False`` (default), a scalar default gain value is used.
    metadata : dict, optional
        Optional metadata overrides to apply on top of
        ``default_parameters_dictionary`` when querying CRDS. This should be
        structured like ``ImageModel.meta`` (i.e., nested dict-like keys).

    Attributes
    ----------
    gain : float or numpy.ndarray
        Gain value(s). This is either a scalar (default) or a 2D per-pixel array
        read from the CRDS reference file.
    usecrds : bool
        Whether CRDS reference files are used.
    metadata : dict or None
        Stored metadata overrides used for CRDS lookup.
    """

    def __init__(self, usecrds=False, metadata=None):
        self.gain = gain
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata)

    def _get_crds_model(self, metadata=None):
        """Populate ``self.gain`` from the CRDS gain reference file.

        This method builds a minimal Roman ``ImageModel`` to obtain CRDS
        parameters, applies default metadata from
        ``default_parameters_dictionary``, and optionally applies caller-provided
        metadata overrides. It then requests the ``gain`` reference type from
        CRDS and reads the gain map from the returned ASDF file.

        Parameters
        ----------
        metadata : dict, optional
            Metadata overrides to apply before CRDS lookup. If provided, values
            are merged into the model metadata tree.

        Notes
        -----
        The gain map stored in the reference file may include border pixels.
        These are removed by slicing ``nborder`` pixels from each edge.
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
            reftypes=["gain"],
            observatory="roman",
        )

        print(ref_file)

        with asdf.open(ref_file["gain"]) as f:
            self.gain = f["roman"]["data"][
                nborder:-nborder, nborder:-nborder
            ].copy()

    def apply(self, img):
        """Apply the gain correction to an image (in place).

        The operation performed is::

            img = img / gain

        which is commonly used to convert an image in electrons (e-) to data
        numbers (DN) when the gain is expressed in e-/DN.

        Parameters
        ----------
        img : galsim.Image or numpy.ndarray
            Image to be gain-corrected. If a :class:`galsim.Image` is provided,
            the underlying ``img.array`` is modified in place. If a NumPy array
            is provided, the array is modified in place (when possible).

        Returns
        -------
        None
            The input is modified in place. (If you need a copy, pass in a copy
            of the array or image.)
        """
        if isinstance(img, galsim.Image):
            img_arr = img.array
        else:
            img_arr = img
        img_arr /= self.gain
        if isinstance(img, galsim.Image):
            img.array = img_arr
        else:
            img[:] = img_arr
