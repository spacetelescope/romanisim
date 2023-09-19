"""Miscellaneous utility routines for the simulation file maker scripts.
"""

from copy import deepcopy
import asdf
from astropy import table
from astropy import time
from astropy import coordinates
import galsim
from galsim import roman
from romanisim import catalog, image, wcs
from romanisim import parameters
from romanisim import log

def merge_nested_dicts(dict1, dict2):
    """
    Merge two nested dictionaries.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to be merged. This dict receives the merged dictionary.

    dict2 : dict
        Second dictionary to be merged.

    Returns
    -------
    None
        dict 1 is updated with the merge output.

    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_nested_dicts(dict1[key], value)
        else:
            dict1[key] = value



def set_metadata(meta=None, date=None, bandpass='F087', sca=7, ma_table_number=1):
    """
    Set / Update metadata parameters

    Parameters
    ----------
    meta : dict
        Dictionary containing all the metadata for a roman image.
    date : str
        UTC format string containing the date and time
    bandpass : str
        String containing the optical element to simulate
    sca : int
        Integer identifier of the detector to simulate (starting at 1)
    ma_table_number : int
        Integer specifying which MA Table entry to use

    Returns
    -------
    meta : dict
        Dictionary created / updated for the above parameters.
    """

    # Set up default metadata if undefined
    if meta is None:
        meta = deepcopy(parameters.default_parameters_dictionary)

    # Set start time, if applicable
    if date is not None:
        meta['exposure']['start_time'] = time.Time(date, format='isot')

    # Detector
    meta['instrument']['detector'] = f'WFI{sca:02d}'

    # Observational metadata
    meta['instrument']['optical_element'] = bandpass
    meta['exposure']['ma_table_number'] = ma_table_number

    return meta


def create_catalog(metadata=None, catalog_name=None, bandpasses=['F087'],
                   rng=None, nobj=1000, usecrds=True,
                   coord=(roman.n_pix / 2, roman.n_pix / 2), radius=0.01):
    """
    Create catalog object.

    Parameters
    ----------
    metadata : dict
        Dictionary containing all the metadata for a roman image.
    catalog_name : str
        Catalog file name (to make a catalog object from a file)
    bandpasses : list
        List of optical elements to simulate
    rng : galsim.random.UniformDeviate
        Uniform distribution based off of a random seed
    nobj : int
        Number of objects to simulate
    usecrds : bool
        Switch to look to use reference files matched in CRDS
    coord : tuple, or SkyCoord
        location at which to generate catalog
        If around a particular location on the sky, a SkyCoord,
        otherwise a tuple (x, y) with the desired pixel coordinates.
    x : float or quantity
        X [float] or RA [quantity] position at the center to simulate
    y : float or quantity
        Y [float] or DEC [quantity] position at the center to simulate
    radius : float
        Radius to simulate object in (only used for RA/DEC quantities)

    Returns
    -------
    cat : astropy.table.table.Table
        Table containing object data for simulation.

    """
    if catalog_name is None and metadata is None:
        raise ValueError('Must set either catalog_name or metadata')

    # Create catalog
    if catalog_name is None:
        # Create a catalog from scratch
        # Create wcs object
        twcs = wcs.get_wcs(metadata, usecrds=usecrds)

        if not isinstance(coord, coordinates.SkyCoord):
            coord = twcs.toWorld(galsim.PositionD(*coord))

        cat = catalog.make_dummy_table_catalog(
            coord, bandpasses=bandpasses, nobj=nobj, rng=rng)
    else:
        log.warning('Catalog input will probably not work unless the catalog '
                    'covers a lot of area or you have thought carefully about '
                    'the relation between the boresight and the SCA locations.')
        cat = table.Table.read(catalog_name, comment="#", delimiter=" ")

    return cat


def simulate_image_file(args, metadata, cat, rng=None, persist=None):
    """
    Simulate an image and write it to a file.

    Parameters
    ----------
    args : argparse.Namespace
        Argument object containing variables specified for image simulation.
    metadata : dict
        Dictionary containing all the metadata for a roman image.
    cat : astropy.table.table.Table
        Table containing object data for simulation.
    rng : galsim.random.UniformDeviate
        Uniform distribution based off of a random seed
    persist : romanisim.persistence.Persistence
        Persistence object

    Returns
    -------

    """

    # Simulate image
    im, extras = image.simulate(
        metadata, cat, usecrds=args.usecrds,
        webbpsf=args.webbpsf, level=args.level, persistence=persist,
        rng=rng)

    # Create metadata for simulation parameter
    romanisimdict = deepcopy(vars(args))
    if 'filename' in romanisimdict:
        romanisimdict['filename'] = str(romanisimdict['filename'])
    romanisimdict.update(**extras)

    # Write file
    af = asdf.AsdfFile()
    af.tree = {'roman': im, 'romanisim': romanisimdict}

    af.write_to(args.filename)
