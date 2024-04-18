"""Miscellaneous utility routines for the simulation file maker scripts.
"""

from copy import deepcopy
import os
import re
import defusedxml.ElementTree
import numpy as np
import asdf
from astropy import table
from astropy import time
from astropy import coordinates
import galsim
from galsim import roman
import roman_datamodels
from roman_datamodels import stnode
from romanisim import catalog, image, wcs
from romanisim import parameters, log

NMAP = {'apt':'{http://www.stsci.edu/Roman/APT}'}

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


def set_metadata(meta=None, date=None, bandpass='F087', sca=7,
                 ma_table_number=1, truncate=None):
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
    meta['exposure']['read_pattern'] = parameters.read_pattern[ma_table_number]
    if truncate is not None:
        meta['exposure']['read_pattern'] = meta['exposure']['read_pattern'][:truncate]
        meta['exposure']['truncated'] = True
    else:
        meta['exposure']['truncated'] = False

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
        distortion_file = parameters.reference_data["distortion"]
        if distortion_file is not None:
            dist_model = roman_datamodels.datamodels.DistortionRefModel(distortion_file)
            distortion = dist_model.coordinate_distortion_transform
        else:
            distortion = None
        twcs = wcs.get_wcs(metadata, usecrds=usecrds, distortion=distortion)

        if not isinstance(coord, coordinates.SkyCoord):
            coord = twcs.toWorld(galsim.PositionD(*coord))

        cat = catalog.make_dummy_table_catalog(
            coord, bandpasses=bandpasses, nobj=nobj, rng=rng)
    else:
        cat = table.Table.read(catalog_name, comment="#", delimiter=" ")
        bandpass = [f for f in cat.dtype.names if f[0] == 'F']
        bad = np.zeros(len(cat), dtype='bool')
        for b in bandpass:
            bad |= ~np.isfinite(cat[b])
            if hasattr(cat[b], 'mask'):
                bad |= cat[b].mask
        cat = cat[~bad]
        nbad = np.sum(bad)
        if nbad > 0:
            log.info(f'Removing {nbad} catalog entries with non-finite or '
                     'masked fluxes.')

    return cat


def parse_filename(filename):
    """
    Pry program / pass / visit / ... information out of the filename.

    Parameters
    ----------
    filename : str
        filename to parse

    Returns
    -------
    dictionary of metadata, or None if filename is non-standard
    """

    # format is:
    # r + PPPPPCCAAASSSOOOVVV_ggsaa_eeee_DET_suffix.asdf
    # PPPPP = program
    # CC = execution plan number
    # AAA = pass number
    # SSS = segment number
    # OOO = observation number
    # VVV = visit number
    # gg = group identifier
    # s = sequence identifier
    # aa = activity identifier
    # eeee = exposure number
    # rPPPPPCCAAASSSOOOVVV_ggsaa_eeee
    # 0123456789012345678901234567890
    if len(filename) < 31:
        return None

    regex = (r'r(\d{5})(\d{2})(\d{3})(\d{3})(\d{3})(\d{3})'
              '_(\d{2})(\d{1})([a-zA-Z0-9]{2})_(\d{4})')
    pattern = re.compile(regex)
    filename = filename[:31]
    match = pattern.match(filename)
    if match is None:
        return None
    out = dict(obs_id=filename.replace('_', '')[1:],
               visit_id=filename[1:20],
               program=match.group(1),  # this one is a string
               execution_plan=int(match.group(2)),
               # pass = int(match.group(3))
               segment=int(match.group(4)),
               observation=int(match.group(5)),
               visit=int(match.group(6)),
               visit_file_group=int(match.group(7)),
               visit_file_sequence=int(match.group(8)),
               visit_file_activity=match.group(9),  # this one is a string
               exposure=int(match.group(10)))
    out['pass'] = int(match.group(3))
    # not done above because pass is a reserved python keyword
    return out


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

    basename = os.path.basename(args.filename)
    obsdata = parse_filename(basename)
    if obsdata is not None:
        im['meta']['observation'].update(**obsdata)
    im['meta']['filename'] = stnode.Filename(basename)

    pretend_spectral = getattr(args, 'pretend_spectral', None)
    if pretend_spectral is not None:
        im['meta']['exposure']['type'] = (
            'WFI_' + args.pretend_spectral.upper())
        im['meta']['instrument']['optical_element'] = (
            args.pretend_spectral.upper())

    drop_extra_dq = getattr(args, 'drop_extra_dq', False)
    if drop_extra_dq and ('dq' in romanisimdict):
        romanisimdict.pop('dq')

    # Write file
    af = asdf.AsdfFile()
    af.tree = {'roman': im, 'romanisim': romanisimdict}

    af.write_to(open(args.filename, 'wb'))


def parse_apt_file(filename, csv_exposures):
    """
    Pry program / pass / visit / ... information out of the apt file.

    Parameters
    ----------
    filename : str
        filename of apt to parse

    csv_exposures : int
        number of exposures specified by the apt file

    Returns
    -------
    list of name prefixes
    """

    # format is:
    # r + PPPPPCCAAASSSOOOVVV_ggsaa_eeee_DET_suffix.asdf
    # PPPPP = program
    # CC = execution plan number
    # AAA = pass number
    # SSS = segment number
    # OOO = observation number
    # VVV = visit number
    # gg = group identifier
    # s = sequence identifier
    # aa = activity identifier
    # eeee = exposure number
    # rPPPPPCCAAASSSOOOVVV_ggsaa_eeee

    # Parse the xml
    apt_tree = defusedxml.ElementTree.parse(filename)
    program = apt_tree.find('.//{*}ProgramID', namespaces=NMAP).text \
        if apt_tree.find('.//{*}ProgramID', namespaces=NMAP).text else 1

    execution_plan = 1
    pass_plan_tree = apt_tree.find('.//{*}PassPlans', namespaces=NMAP)
    pass_plans = pass_plan_tree.findall('.//{*}PassPlan', namespaces=NMAP) \
        if pass_plan_tree.findall('.//{*}PassPlan', namespaces=NMAP) else [-1]

    total_apt_exposures = 0
    for exp in apt_tree.findall('.//{*}NumberOfExposures', namespaces=NMAP):
        total_apt_exposures += int(exp.text)

    # Account for extra exposures due to dither pattern
    if csv_exposures > total_apt_exposures:
        avg_visits = int(csv_exposures / total_apt_exposures)
    else:
        avg_visits = 1

    name_prefix_lst = []

    # Set defaults
    segment = 1
    group = 1
    sequence = 1
    activity = 1

    for pn in pass_plans:
        pass_number = pn.get('Number')
        obs_num = 0      

        for on in pn.findall('.//{*}Observation', namespaces=NMAP):
            obs_num += 1
            exp_num = int(on.find('.//{*}NumberOfExposures', namespaces=NMAP).text)

            for vn in range(avg_visits):
                for en in range(exp_num):
                    # Can make the Programmatic observation identifier list now
                    prefix = "r" + str(program).zfill(5) + str(execution_plan).zfill(2) + str(pass_number).zfill(3) +\
                             str(segment).zfill(3) + str(obs_num).zfill(3) + str(vn+1).zfill(3) + "_" +\
                             str(group).zfill(2) + str(sequence).zfill(1) + str(activity).zfill(2) + "_" + str(en+1).zfill(4)
                    name_prefix_lst.append(prefix)
    
    return name_prefix_lst
