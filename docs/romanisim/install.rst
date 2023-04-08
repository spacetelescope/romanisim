Installation
============

To install ::

    pip install romanisim

and you should be largely set!

There are a few dependencies that may cause more difficulty.  First,
`WebbPSF <https://webbpsf.readthedocs.io>`_ requires data files to
operate.  See the `docs
<https://webbpsf.readthedocs.io/en/latest/installation.html#installing-the-required-data-files>`_
for instructions on obtaining the relevant data files and pointing the
``WEBBPSF_PATH`` environment variable to them.  This issue can be
avoided by not setting the ``--webbpsf`` argument, in which case
``romanisim`` uses the GalSim modeling of the Roman PSF.

Second, some synthetic scene generation tools use images of galaxies
distributed separately from the main GalSim source.  See `here
<https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog>`_
for information on obtaining the COSMOS galaxies for use with GalSim.
The ``romanisim`` package also has a less sophisticated scene modeling
toolkit, which just renders Sersic galaxies.  The command line
interface to ``romanisim`` presently uses supports Sersic galaxy
rendering, and so many users may not need to download the COSMOS galaxies.

Third, ``romanisim`` can work with the Roman `CRDS
<https://github.com/spacetelescope/crds>`_ system.  This functionality
is not available to the general community at the time of writing.
Using CRDS requires specifying the ``CRDS_PATH`` and
``CRDS_SERVER_URL`` variables.  CRDS is not used unless the
``--usecrds`` argument is specified; do not include this argument
unless you have access to the Roman CRDS.
