"""Very simple APT reader.

Converts an APT file into a list of (ra, dec, angle, filter, date,
exposure time) needed for generating observations.  This is adequate
for reading in a few of the example Roman APTs but only supports a tiny
fraction of what an APT file seems able to do.
"""

import xml
from xml.etree import ElementTree
from astropy import coordinates
from astropy import units as u
import dataclasses
import datetime

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
    exptime: float
    date: datetime.datetime


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
    tree = ElementTree.parse(filename)
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
        targ = target_dict[int(o.find(XMLNS + 'Target').text)]
        bandpass = o.find(XMLNS + 'OpticalElement').text
        exptime = -1  # Where is this located?
        date = datetime.datetime(2000, 1, 1)  # How should I derive this?
        obs = Observation(targ, bandpass, exptime, date)
        obslist.append(obs)
    return obslist
