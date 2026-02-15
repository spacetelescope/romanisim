import asdf
import crds
import galsim
import numpy as np
import roman_datamodels

from scipy import ndimage

from .parameters import default_parameters_dictionary

__all__ = ["IPC"]

# # Default IPC kernel
# # IPC kernel is unnormalized at first.  We will normalize it.
# ipc_kernel = np.array(
#     [
#         [0.001269938, 0.015399776, 0.001199862],
#         [0.013800177, 1.0, 0.015600367],
#         [0.001270391, 0.016129619, 0.001200137],
#     ]
# )
# ipc_kernel /= np.sum(ipc_kernel)

# from draft wfisim documentation
# assumed to be compatible in direction with scipy.ndimage.convolve.
# this is consistent with Andrea Bellini's convention, where, in the
# following kernel, 0.2% of the flux is redistributed to the pixel
# that is +1 - N spaces ahead in memory.
ipc_kernel = np.array(
    [[0.21, 1.66, 0.22], [1.88, 91.59, 1.87], [0.21, 1.62, 0.2]]
)
ipc_kernel /= np.sum(ipc_kernel)


class IPC(object):
    """Inter-pixel capacitance (IPC) convolution model.

    Parameters
    ----------
    usecrds : bool, optional
        If True, load the IPC kernel from CRDS using the provided
        ``metadata`` (default: False). If False, use the built-in
        default kernel defined in this module.
    metadata : dict or None, optional
        Optional metadata overrides applied to the data model before
        CRDS lookup.

    Attributes
    ----------
    ipc_kernel : numpy.ndarray
        The normalized 2D IPC convolution kernel.

    Notes
    -----
    The kernel is applied with ``scipy.ndimage.convolve``. The kernel sum
    determines the DC gain of the operation; kernels loaded from CRDS are
    explicitly normalized to unit sum.
    """

    def __init__(self, usecrds=False, metadata=None):
        self.usecrds = usecrds
        self.metadata = metadata
        if self.usecrds:
            self._get_crds_model(metadata=self.metadata)
        else:
            self.ipc_kernel = ipc_kernel

    def _get_crds_model(self, metadata=None):
        """Load and normalize the IPC kernel from CRDS.

        This method constructs a Roman ImageModel, populates
        it with default parameters and any user-provided ``metadata``
        overrides, and then queries CRDS for the ``ipc`` reference file.

        Parameters
        ----------
        metadata : dict or None, optional
            Metadata overrides applied to the data model before CRDS lookup.

        Notes
        -----
        The kernel read from the reference file is normalized in-place so
        that ``ipc_kernel.sum() == 1``.
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
            reftypes=["ipc"],
            observatory="roman",
        )

        print(ref_file)

        with asdf.open(ref_file["ipc"]) as f:
            self.ipc_kernel = f["roman"]["data"]
            self.ipc_kernel /= np.sum(self.ipc_kernel)

    def apply(self, img, edge_treatment="constant", fill_value=0.0):
        """Apply IPC convolution to an image (in-place).

        Parameters
        ----------
        img : numpy.ndarray or galsim.Image
            The image to convolve. If a ``galsim.Image`` is provided, its
            ``.array`` will be updated. If a NumPy array is provided, the
            input array is overwritten via ``img[:] = ...``.
        edge_treatment : {"constant","nearest","reflect","mirror","wrap"}, optional
            Boundary handling mode passed to ``scipy.ndimage.convolve``
            (default: ``'constant'``).
        fill_value : float, optional
            Fill value used when ``edge_treatment='constant'`` (default: 0.0).

        Returns
        -------
        None
            The result is written back into ``img``.
        """
        if isinstance(img, galsim.Image):
            img_arr = img.array
        else:
            img_arr = img

        if img_arr.ndim == 2:
            img_arr = ndimage.convolve(
                img_arr, self.ipc_kernel, mode=edge_treatment, cval=fill_value
            )
        elif img_arr.ndim == 3:  # Convolution on 3D resultants
            img_arr = ndimage.convolve(
                img_arr,
                self.ipc_kernel[None, ...],
                mode=edge_treatment,
                cval=fill_value,
            )
        if isinstance(img, galsim.Image):
            img.array = img_arr
        else:
            img = img_arr
