"""Miscellaneous utility routines for the simulation file maker scripts.
"""

from copy import deepcopy
import os
import pathlib
import re
import defusedxml.ElementTree
import asdf
from astropy import time
from astropy import coordinates
from astropy import units as u
import galsim
from galsim import roman
import roman_datamodels
from romanisim import catalog, image, wcs
from romanisim import parameters, log
from romanisim.util import calc_scale_factor
import romanisim
import crds
from crds.client import api


NMAP = {'apt': 'http://www.stsci.edu/Roman/APT'}


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
                 ma_table_number=4, truncate=None, scale_factor=1.0, usecrds=False):
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
    scale_factor : float
        Velocity aberration-induced scale factor
    usecrds : bool
        Use CRDS to get MA table reference file

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
    if usecrds:
        try:
            context = api.get_default_context('roman')
        except crds.ServiceError:
            context = None
        ref = crds.getreferences({'ROMAN.META.INSTRUMENT.NAME': 'wfi', 'ROMAN.META.EXPOSURE.START_TIME': meta['exposure']['start_time'].value}, reftypes=['matable'], context=context, observatory='roman')
        matab_file = ref['matable']
        matab = asdf.open(matab_file)

        parameters.ma_table_reference = matab

        meta['exposure']['read_pattern'] = matab['roman']['science_tables'][f'SCI{ma_table_number:04}']['science_read_pattern']
    else:
        meta['exposure']['read_pattern'] = parameters.read_pattern[ma_table_number]

    if truncate is not None:
        meta['exposure']['read_pattern'] = meta['exposure']['read_pattern'][:truncate]
        meta['exposure']['truncated'] = True
    else:
        meta['exposure']['truncated'] = False

    meta['exposure']['nresultants'] = len(meta['exposure']['read_pattern'])

    # Velocity aberration
    if scale_factor <= 0.:
        scale_factor = calc_scale_factor(meta['exposure']['start_time'], meta['wcsinfo']['ra_ref'], meta['wcsinfo']['dec_ref'])
    meta['velocity_aberration']['scale_factor'] = scale_factor

    # Fill out some ephemeris information, presuming all is earth.
    position, velocity = coordinates.get_body_barycentric_posvel('earth', meta['exposure']['start_time'])
    position = position.xyz.to(u.km)
    velocity = velocity.xyz.to(u.km / u.s)
    meta['ephemeris']['time'] = meta['exposure']['start_time'].mjd
    meta['ephemeris']['spatial_x'] = position.value[0]
    meta['ephemeris']['spatial_y'] = position.value[1]
    meta['ephemeris']['spatial_z'] = position.value[2]
    meta['ephemeris']['velocity_x'] = velocity.value[0]
    meta['ephemeris']['velocity_y'] = velocity.value[1]
    meta['ephemeris']['velocity_z'] = velocity.value[2]

    return meta


def create_catalog(metadata=None, catalog_name=None, bandpasses=['F087'],
                   rng=None, nobj=1000, usecrds=True,
                   coord=None, radius=0.1):
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
        None does the center of the SCA.
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

    if coord is None:
        coord = (roman.n_pix / 2, roman.n_pix / 2)

    distortion_file = parameters.reference_data["distortion"]
    if distortion_file is not None:
        dist_model = roman_datamodels.datamodels.DistortionRefModel(distortion_file)
        distortion = dist_model.coordinate_distortion_transform
    else:
        distortion = None

    if metadata is not None:
        twcs = wcs.get_wcs(metadata, usecrds=usecrds, distortion=distortion)
        if isinstance(coord, galsim.CelestialCoord):
            coord = coord
        elif not isinstance(coord, coordinates.SkyCoord):
            coord = twcs.toWorld(galsim.PositionD(*coord))

    # Create catalog
    if catalog_name is None:
        # Create a catalog from scratch
        # Create wcs object

        cat = catalog.make_dummy_table_catalog(
            coord, bandpasses=bandpasses, nobj=nobj, rng=rng, cosmos=True)
    else:
        # Set date
        if metadata:
            # Level 1 or 2
            if "exposure" in metadata:
                date = metadata['exposure']['start_time']
            else:
                # Level 3
                date = time.Time(metadata['coadd_info']['time_mean'], format='mjd')
        else:
            date = None

        cat = catalog.read_catalog(catalog_name, coord, date=date,
                                   radius=radius, bandpasses=bandpasses)

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
    # r + PPPPPCCAAASSSOOOVVV_eeee_DET_suffix.asdf
    # PPPPP = program
    # CC = execution plan number
    # AAA = pass number
    # SSS = segment number
    # OOO = observation number
    # VVV = visit number
    # eeee = exposure number
    # rPPPPPCCAAASSSOOOVVV_eeee
    # 0123456789012345678901234
    if len(filename) < 31:
        return None

    regex = (r'r(\d{5})(\d{2})(\d{3})(\d{3})(\d{3})(\d{3})'
             r'_(\d{4})')
    pattern = re.compile(regex)
    filename = filename[:25]
    match = pattern.match(filename)
    if match is None:
        return None
    out = dict(observation_id=filename.replace('_', '')[1:],
               visit_id=filename[1:20],
               program=int(match.group(1)),  # this one is a string
               execution_plan=int(match.group(2)),
               # pass = int(match.group(3))
               segment=int(match.group(4)),
               observation=int(match.group(5)),
               visit=int(match.group(6)),
               exposure=int(match.group(7)))
    out['pass'] = int(match.group(3))
    # not done above because pass is a reserved python keyword
    return out


def format_filename(filename, sca, bandpass=None, pretend_spectral=None):
    """Add SCA and filter information to a filename.

    This parameter turns a string like out_{}_{bandpass}.asdf into
    out_wfi01_f184.asdf.  It differs from format(...) calls in that it won't
    choke if the target filename includes no {} symbols to format.

    If pretend_spectral is set, the bandpass portion is filled out with the
    value of pretend_spectral rather than the bandpass, so that
    out_{}_{bandpass}.asdf becomes out_wfi01_grism.asdf, for example, if
    pretend_bandpass is grism and bandpass is f158.  This is to support
    creation of files that have metadata that look like spectroscopic files
    while the actual pixels are simulated with a different optical bandpass.

    Parameters
    ----------
    filename : str
        file name to format
    sca : int
        detector to insert into filename
    bandpass : str
        optical element to insert into filename
    pretend_spectral : str
        pretend that optical observation was made in this spectroscopic mode
    """
    args = []
    kwargs = dict()
    pname = pathlib.Path(filename)
    bname = pname.parts[-1]
    if '{}' in bname:
        args.append(f'wfi{sca:02d}')
    if '{bandpass}' in bname:
        bp = bandpass if pretend_spectral is None else pretend_spectral
        kwargs['bandpass'] = bp.lower()
    return pname.with_name(bname.format(*args, **kwargs))


def simulate_image_file(args, metadata, cat, rng=None, persist=None, psf_keywords=dict(), **kwargs):
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
    psf_keywords : dict
        Keywords passed to the PSF generation routine. 
        For STPSF, this dict can also include an "stpsf_options" dictionary to specify WFI object options (e.g. defocus, jitter).
    """
    if getattr(args, 'webbpsf', False) or getattr(args, 'stpsf', False):
        log.warning('Warning: webbpsf and stpsf arguments are deprecated, please use '
                    '"--psftype stpsf" instead.')
        del args.stpsf
        del args.webbpsf
        args.psftype = 'stpsf'

    filename = format_filename(args.filename, args.sca, bandpass=args.bandpass,
                               pretend_spectral=args.pretend_spectral)

    # Simulate image
    im, extras = image.simulate(
        metadata, cat, usecrds=args.usecrds,
        psftype=args.psftype, level=args.level, persistence=persist,
        rng=rng, psf_keywords=psf_keywords, **kwargs)

    # Create metadata for simulation parameter
    romanisimdict = deepcopy(vars(args))
    if 'filename' in romanisimdict:
        romanisimdict['filename'] = str(romanisimdict['filename'])
    romanisimdict.update(**extras)
    romanisimdict['version'] = romanisim.__version__

    basename = os.path.basename(filename)
    obsdata = parse_filename(basename)
    if obsdata is not None:
        im['meta']['observation'].update(**obsdata)
    im['meta']['observation']['visit_file_group'] = 0
    im['meta']['observation']['visit_file_sequence'] = 1
    im['meta']['observation']['visit_file_activity'] = '01'
    im['meta']['filename'] = basename

    pretend_spectral = getattr(args, 'pretend_spectral', None)
    if pretend_spectral is not None:
        im['meta']['exposure']['type'] = 'WFI_SPECTRAL'
        im['meta']['instrument']['optical_element'] = (
            args.pretend_spectral.upper())
        gs = im['meta']['guide_star']
        if 'window_xstart' in gs:
            gs['window_xstop'] = gs['window_xstart'] + 170
            gs['window_ystop'] = gs['window_ystart'] + 24

    drop_extra_dq = getattr(args, 'drop_extra_dq', False)
    if drop_extra_dq and ('dq' in romanisimdict):
        romanisimdict.pop('dq')

    # Write file
    af = asdf.AsdfFile()
    af.tree = {'roman': im, 'romanisim': romanisimdict}

    af.write_to(open(filename, 'wb'))


def parse_apt_file(filename):
    """
    Extract metadata from apt file and put it into our preferred structure.

    Parameters
    ----------
    filename : str
        filename of apt to parse

    Returns
    -------
    dictionary of metadata
    """

    metadata = dict()

    keys = [(('ProgramInformation', 'Title'), ('program', 'title')),
            (('ProgramInformation', 'PrincipalInvestigator',
              'InvestigatorAddress', 'LastName'),
             ('program', 'pi_name')),
            (('ProgramInformation', 'ProgramCategory'),
             ('program', 'category')),
            (('ProgramInformation', 'ProgramCategorySubtype'),
             ('program', 'subcategory')),
            (('ProgramInformation', 'ProgramID'), ('observation', 'program'))]

    def get_apt_key(tree, keypath):
        element = tree.find('apt:' + keypath[0], namespaces=NMAP)
        for key in keypath[1:]:
            element = element.find('apt:' + key, namespaces=NMAP)
        if element is not None:
            out = element.text
        else:
            out = ''
            log.info('Could not find key in apt file: ' + str(keypath))
        return out

    def update_metadata(metadata, keypath, value):
        d = metadata
        for key in keypath[:-1]:
            if key not in metadata:
                metadata[key] = dict()
            d = metadata[key]
        d[keypath[-1]] = value

    tree = defusedxml.ElementTree.parse(filename)

    for aptkey, metadatakey in keys:
        value = get_apt_key(tree, aptkey)
        update_metadata(metadata, metadatakey, value)
        # only works because these are all strings at present

    metadata['observation']['program'] = (
        f'{int(metadata["observation"]["program"]):05d}')
    # hack to get the program to have the 0 prefixes.

    return metadata
