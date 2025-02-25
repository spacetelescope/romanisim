"""romanisim: a Roman image simulator tool based on galsim.

romanisim aims to use galsim as a tool to implement robust, accurate
simulations of Roman WFI observations.  It tries to use official
Roman tools whenever possible: e.g., for implementing the WCS
distortion model, bandpass, and point spread function.

It also aims to simulate the process by which individual reads of
the detector are averaged into resultants before eventually producing
rate images that are analagous to typical astronomical images.
"""
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from importlib.metadata import version
import logging

log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log.setLevel('INFO')
# doesn't feel like it should be necessary, but it seems we don't get some
# test log messages without setting setLevel('INFO') after configuring
# with the basicConfig.

__version__ = version(__name__)
