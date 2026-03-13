import crds
import galsim
import numpy as np
from roman_datamodels import datamodels

from .parameters import default_parameters_dictionary, nborder

__all__ = ["Saturation"]


class Saturation(object):
    """
    Apply detector saturation to an image.

    This class clips pixel values to a saturation threshold, either using
    a scalar saturation level or a per-pixel saturation map loaded from
    Roman CRDS reference files.

    Parameters
    ----------
    usecrds : bool, optional
        If True, load a per-pixel saturation map from Roman CRDS reference
        files. If False, use a scalar saturation level.
    metadata : dict, optional
            Metadata overrides to apply before CRDS lookup. If provided, values
            are merged into the model metadata tree.
    saturation_level : float, optional
        Scalar saturation level used when `usecrds=False`. Units are assumed
        to match the image pixel values (typically electrons or DN).

    Attributes
    ----------
    usecrds : bool
        Whether CRDS is used to obtain a per-pixel saturation map.
    dq : numpy.ndarray
        Only present if ``usecrds=True`` and ``getdq=True``. The DQ array read
        from the CRDS dark reference, cropped by ``nborder``.
    metadata : dict or None
        Metadata overrides for CRDS lookup.
    saturation_level : float or numpy.ndarray
        Saturation threshold. Scalar if `usecrds=False`, otherwise a 2D array
        with per-pixel saturation limits.
    """

    def __init__(
        self,
        usecrds=False,
        getdq=False,
        metadata=None,
        image_mod=None, 
        reffiles=None, 
        saturation_level=300000,
    ):
        self.usecrds = usecrds
        self.metadata = metadata
        self.saturation_level = saturation_level
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata, image_mod=image_mod, reffiles=reffiles, getdq=getdq)

    def _get_crds_model(self, getdq=False, metadata=None, image_mod=None, reffiles=None):
        """
        Load the Roman saturation reference file from CRDS.

        This method:
          1) Creates a fake Roman ImageModel.
          2) Populates its metadata using `default_parameters_dictionary`.
          3) Applies any user-provided metadata overrides.
          4) Queries CRDS for the Roman "saturation" reference file.
          5) Loads the saturation map and trims detector borders.

        The resulting saturation map is stored in `self.saturation_level`
        as a 2D NumPy array.

        Parameters
        ----------
        getdq : bool, optional
            If True, also read ``roman/dq`` from the dark reference and store it
            as ``self.dq``.
        metadata : dict, optional
            Metadata overrides to apply before CRDS lookup. If provided, values
            are merged into the model metadata tree.
        """

        if image_mod is not None:
            ref_file = crds.getreferences(
                image_mod.get_crds_parameters(),
                reftypes=["saturation"],
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
                reftypes=["saturation"],
                observatory="roman",
            )
        
        if isinstance(ref_file['saturation'], str):
            model = datamodels.open(ref_file['saturation'])
            self.saturation_level = model.data[nborder:-nborder, nborder:-nborder].copy()
            if getdq:
                self.dq = model.dq[nborder:-nborder, nborder:-nborder].copy()

    def apply(self, img):
        """
        Apply saturation clipping to an image (in place).

        Pixel values are clipped to the range
        ``[0, self.saturation_level]`` using `numpy.clip`.

        Parameters
        ----------
        img : galsim.Image or array-like (2D)
            Input image to be modified in place. If a `galsim.Image` is
            provided, its internal array is updated. If a NumPy-like
            array is provided, it is modified directly.

        Notes
        -----
        - This method modifies the input image in place and returns None.
        - When `usecrds=True`, `self.saturation_level` is a 2D array and
          clipping is performed element-wise.
        - Negative values are also clipped to zero.

        Returns
        -------
        None
            The input image is modified in place.
        """
        if isinstance(img, galsim.Image):
            img_arr = img.array
        else:
            img_arr = img
        img_arr = np.clip(img_arr, 0, self.saturation_level)
        if isinstance(img, galsim.Image):
            img.array = img_arr
        else:
            img[:] = img_arr

        # if not self.usecrds:
        #     saturation_array = np.ones_like(img.array) * self.saturation_level
        #     where_sat = np.where(img.array > saturation_array)
        #     img.array[where_sat] = saturation_array[where_sat]
        # else:
        #     # The CRDS saturation references is in DN
        #     # Resultants exceeding the saturation level are clipped at
        #     # the saturation level and marked as saturated.

        #     # [from roman_imsim] this maybe should be better applied at
        #     # read time? it's not actually clear to me what the right
        #     # thing to do is in detail.
        #     if not isinstance(img, u.Quantity):
        #         img *= u.DN
        #     img = np.clip(img, 0 * u.DN, self.saturation_map, out=img)

        #     # m = resultants >= saturation
        #     # dq[m] |= parameters.dqbits['saturated']
        #     # return resultants, dq

        # return img
