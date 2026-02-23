import crds
import galsim
import numpy as np
from roman_datamodels import datamodels

from .parameters import default_parameters_dictionary, nborder

__all__ = ["ReadNoise"]

# Default read noise value
read_noise = 8.5  # e-


class ReadNoise(object):
    """
    Apply detector read noise to an image.

    This class models Gaussian read noise, either using a scalar RMS value
    or a per-pixel RMS map loaded from Roman CRDS reference files. Noise is
    added in place to the input image.

    Parameters
    ----------
    usecrds : bool, optional
        If True, load a per-pixel read-noise map from Roman CRDS reference
        files. If False, use a scalar RMS read-noise value.
    metadata : dict or None, optional
        Optional metadata overrides used when querying CRDS reference files.
        Keys should correspond to entries in the Roman datamodel `meta`.
    rng : galsim.BaseDeviate or compatible, optional
        Random number generator used to draw Gaussian noise. If provided,
        it is wrapped as a `galsim.GaussianDeviate`.
    seed : int or None, optional
        Seed used to initialize the RNG when `rng` is not provided.

    Attributes
    ----------
    read_noise : float or numpy.ndarray
        RMS read noise in electrons per pixel. Scalar if `usecrds=False`,
        otherwise a 2D array with per-pixel RMS values.
    usecrds : bool
        Whether CRDS is used to obtain the read-noise reference.
    metadata : dict or None
        Metadata overrides for CRDS lookup.
    rng : galsim.GaussianDeviate
        RNG used to generate Gaussian noise.
    """

    def __init__(self, usecrds=False, metadata=None, image_mod=None, reffiles=None, rng=None, seed=None):
        self.read_noise = read_noise
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata, image_mod=image_mod, reffiles=reffiles)

        if rng is None and seed is None:
            self.seed = 45
        if rng is None:
            self.rng = galsim.GaussianDeviate(seed)
        else:
            self.rng = galsim.GaussianDeviate(rng)

    def _get_crds_model(self, metadata=None, image_mod=None, reffiles=None):
        """
        Load the Roman read-noise reference file from CRDS.

        This method:
          1) Creates a fake Roman ImageModel.
          2) Populates its metadata using `default_parameters_dictionary`.
          3) Applies any user-provided metadata overrides.
          4) Queries CRDS for the Roman "readnoise" reference file.
          5) Loads the read-noise map and trims detector borders.

        The resulting read-noise map is stored in `self.read_noise` as a
        2D NumPy array.

        Parameters
        ----------
        metadata : dict, optional
            Metadata overrides to apply before CRDS lookup. If provided, values
            are merged into the model metadata tree.

        """

        if image_mod is not None:
            ref_file = crds.getreferences(
                image_mod.get_crds_parameters(),
                reftypes=["dark", "gain"],
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
                reftypes=["readnoise"],
                observatory="roman",
            )
        
        if isinstance(ref_file['readnoise'], str):
            model = datamodels.open(ref_file['readnoise'])
            self.read_noise = model.data[nborder:-nborder, nborder:-nborder].copy()

    def apply(self, img, n_reads=1.0):
        """
        Add Gaussian read noise to an image (in place).

        Noise is drawn from a zero-mean Gaussian distribution with RMS
        ``self.read_noise / sqrt(n_reads)`` and added to the image.

        Parameters
        ----------
        img : galsim.Image or array-like (2D)
            Input image to be modified in place. If a `galsim.Image` is
            provided, its internal array is updated. If a NumPy-like array
            is provided, it is modified directly.
        n_reads : float, optional
            Number of reads contributing to the image. The effective read
            noise is scaled as ``1 / sqrt(n_reads)``.

        Notes
        -----
        - This method modifies the input image in place and returns None.
        - When `usecrds=True`, `self.read_noise` is a 2D array and noise is
          applied per pixel.
        - Pixel units are assumed to be electrons.

        Returns
        -------
        None
            The input image is modified in place.
        """
        if isinstance(img, galsim.Image):
            img_arr = img.array
        else:
            img_arr = img
        noise = np.zeros(img_arr.shape, dtype="f4")
        self.rng.generate(noise)
        noise = noise * self.read_noise / (n_reads**0.5)
        img_arr += noise
        if isinstance(img, galsim.Image):
            img.array = img_arr
        else:
            img[:] = img_arr

        # if not self.usecrds:
        #     gn = galsim.GaussianNoise(self.rng, sigma=self.read_noise)
        #     img.addNoise(gn)
        # else:
        #     # The read noise is averaged down like 1/\sqrt{n_reads},
        #     # where n_reads is the number of reads contributing to
        #     # the resultant.
        #     noise = np.zeros(img.array.shape, dtype="f4")
        #     self.rng.generate(noise)
        #     noise = noise * self.read_noise / (n_reads**0.5)
        #     img.array += noise
        #     return img
