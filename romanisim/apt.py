"""Very simple APT reader.

Converts an APT file into a list of (ra, dec, angle, filter, date,
exposure time) needed for generating observations.  This is adequate
for reading in a few of the example Roman APTs but only supports a tiny
fraction of what an APT file seems able to do.
"""

import defusedxml.ElementTree
from astropy import coordinates
from astropy import units as u
from astropy.table import Table
import dataclasses
import datetime
from romanisim import parameters

XMLNS = '{http://www.stsci.edu/Roman/APT}'


@dataclasses.dataclass
class Target:
    """A target for observation."""
    name: str
    number: int
    coords: coordinates.SkyCoord


@dataclasses.dataclass
class Observation:
    """An observation of a target."""
    target: Target
    bandpass: str
    ma_table: str
    exptime: float
    date: datetime.datetime
    dither_pattern: str
    resultants: int


# Add function to create observations from the dither pattern
def implement_dither(targ, dither_pattern):
    dither_targs = Table(names=('RA', 'DEC', 'PA'))
    obs_ra = targ.coords.ra.value
    obs_dec = targ.coords.dec.value
    obs_pa = 0
    for line in dither_pattern:
        obs_ra += line['RA']
        obs_dec += line['DEC']
        obs_pa += line['PA']
        dither_targs.add_row((obs_ra, obs_dec, obs_pa))

    return dither_targs


def read_apt(filename):
    """Read an APT file, returning a list of observations.

    Parameters
    ----------
    filename : str
        filename of the APT file to read in.

    Returns
    -------
    list[Observation]
        list of Observations in the APT file
    """
    # I don't know anything about reading XML.
    # In general it's very flexible and can do anything.
    # tree = ElementTree.parse(filename)
    tree = defusedxml.ElementTree.parse(filename)
    targs = tree.find(XMLNS + 'Targets')
    target_elements = targs.findall(XMLNS + 'FixedTarget')
    target_dict = dict()
    for target in target_elements:
        t_name = target.find(XMLNS + 'TargetName').text
        t_coords = target.find(XMLNS + 'EquatorialCoordinates').get('Value')
        t_coords = coordinates.SkyCoord(t_coords, unit=(u.hourangle, u.deg))
        t_number = int(target.find(XMLNS + 'Number').text)
        target_dict[t_number] = Target(t_name, t_number, t_coords)
    # I don't see any exposure time information.
    # I do see 'mosaic' information, but I am going to ignore it.
    # I don't see any exposure date information.
    # I think I'm supposed to record the target number, then go through
    # the pass plans, then get the observations in each pass plan.
    passplansection = tree.find(XMLNS + 'PassPlans')
    passplans = passplansection.findall(XMLNS + 'PassPlan')
    observations = sum([p.findall(XMLNS + 'Observation') for p in passplans],
                       [])
    obslist = []
    for o in observations:
        # If Target field is empty, assume only one target and use it
        if o.find(XMLNS + 'Target').text is None:
            targ = target_dict[1]
        else:
            targ = target_dict[int(o.find(XMLNS + 'Target').text)]

        bandpass = o.find(XMLNS + 'OpticalElement').text
        ma_table = o.find(XMLNS + 'MultiAccumTable').text
        exptime = parameters.read_time * parameters.read_pattern[parameters.ma_table_map[ma_table]][-1][-1]
        dither_pattern = o.find(XMLNS + 'Dither').text
        resultants = int(o.find(XMLNS + 'Resultant').text)
        date = datetime.datetime(2000, 1, 1)  # How should I derive this?

        obs = Observation(targ, bandpass, ma_table, exptime, date, dither_pattern, resultants)
        obslist.append(obs)
    return obslist


def create_sim_table(obslist):
    # ("RA", "DEC", "PA", "BANDPASS", "MA_TABLE_NUMBER", "DURATION")
    sim_table = Table(names=parameters.sim_table_names, dtype=parameters.sim_table_dtypes)
    for obs in obslist:
        dither_pattern = parameters.dither_pattern[obs.dither_pattern]
        dithered_targs = implement_dither(obs.target, dither_pattern)
        for t_idx, targ in enumerate(dithered_targs):
            for res in range(obs.resultants):
                sim_table.add_row((dithered_targs["RA"][t_idx],
                                   dithered_targs["DEC"][t_idx],
                                   dithered_targs["PA"][t_idx],
                                   obs.bandpass,
                                   parameters.ma_table_map[obs.ma_table],
                                   obs.exptime))

    return sim_table
