import crds
import warnings
from roman_datamodels import datamodels
from .parameters import default_parameters_dictionary

__all__ = ["get_ref_files"]

def get_ref_files(image_mod=None, metadata=None, reffiles=None, reftypes=[]):
    ref_file = {}
    if image_mod is not None:
        try:
            ref_file = crds.getreferences(
                image_mod.get_crds_parameters(),
                reftypes=reftypes,
                observatory="roman",
            )
        except Exception as e:
            warnings.warn(f"Final CRDS getreferences call failed: {e}")
            return {}
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
        try:
            ref_file = crds.getreferences(
                image_mod.get_crds_parameters(),
                reftypes=reftypes,
                observatory="roman",
            )
        except Exception as e:
            warnings.warn(f"Final CRDS getreferences call failed: {e}")
            return {}
    return ref_file