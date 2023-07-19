"""Miscellaneous utility routines for the simulation file maker scripts.
"""

from copy import deepcopy
import asdf
from astropy.table import Table
from astropy.time import Time
import galsim
from romanisim import catalog, image, wcs
from romanisim.parameters import default_parameters_dictionary
from romanisim import log
from coord.angle import Angle


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
        meta = deepcopy(default_parameters_dictionary)

    # Set start time, if applicable
    if date is not None:
        meta['exposure']['start_time'] = Time(date, format='isot')

    # Detector
    meta['instrument']['detector'] = f'WFI{sca:02d}'

    # Observational metadata
    meta['instrument']['optical_element'] = bandpass
    meta['exposure']['ma_table_number'] = ma_table_number

    return meta


def create_catalog(metadata=None, catalog_name=None, bandpasses=['F087'], rng=None,
                   nobj=1000, usecrds=True,
                   x=galsim.roman.n_pix / 2, y=galsim.roman.n_pix / 2, radius=0.0):
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

    # Create catalog
    if catalog_name is None:
        # Create a catalog from scratch
        # Create wcs object
        twcs = wcs.get_wcs(metadata, usecrds=usecrds)

        # Check for pointing input (for multiple images)
        if isinstance(x, Angle):
            rd = galsim.CelestialCoord(ra=x, dec=y)
            cat = catalog.make_dummy_table_catalog(rd, radius=radius,
                                                   bandpasses=bandpasses, nobj=nobj, rng=rng)
        else:
            # Pixel input (for individual image)
            rd = twcs.toWorld(galsim.PositionD(x, y))
            cat = catalog.make_dummy_table_catalog(rd, bandpasses=bandpasses, nobj=nobj, rng=rng)
    else:
        log.warning('Catalog input will probably not work unless the catalog '
                    'covers a lot of area or you have thought carefully about '
                    'the relation between the boresight and the SCA locations.')
        cat = Table.read(catalog_name, comment="#", delimiter=" ")

    return cat


def simulate_image_file(args, metadata, cat, rng, persist, output=None):
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
    romanisimdict.update(**extras)

    # Write file
    af = asdf.AsdfFile()
    af.tree = {'roman': im, 'romanisim': romanisimdict}

    if output is None:
        af.write_to(args.filename)
    else:
        af.write_to(output)
