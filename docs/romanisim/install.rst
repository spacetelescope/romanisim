Installation
============

To install ::

    pip install romanisim

and you should be largely set!

The most frequently encountered difficulty installing romanisim is
when GalSim is unable to find FFTW.  If your system does not have the
FFTW library, these should be installed before romanisim and GalSim.

Another problematic dependency is `STPSF
<https://stpsf.readthedocs.io>`_, which requires data files to
operate.  See the `docs
<https://stpsf.readthedocs.io/en/latest/installation.html#installing-the-required-data-files>`_
for instructions on obtaining the relevant data files and pointing the
``STPSF_PATH`` environment variable to them.  This issue can be
avoided by not setting the ``--stpsf`` argument, in which case
``romanisim`` uses the GalSim modeling of the Roman PSF.

Additionally, some synthetic scene generation tools use images of galaxies
distributed separately from the main GalSim source.  See `here
<https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog>`_
for information on obtaining the COSMOS galaxies for use with GalSim.
The ``romanisim`` package also has a less sophisticated scene modeling
toolkit, which just renders Sersic galaxies.  The command line
interface to ``romanisim`` presently uses Sersic galaxy
rendering, and so many users may not need to download the COSMOS galaxies.

Finally, ``romanisim`` can work with the Roman `CRDS
<https://github.com/spacetelescope/crds>`_ system.
Using CRDS requires specifying the ``CRDS_PATH`` and
``CRDS_SERVER_URL`` variables.  Using CRDS brings in the latest
reference files and its use is encouraged; specify the
``--usecrds`` argument to enable it.

In summary, the basic install process looks like this::

    pip install romanisim
    # to get a specific version, use instead
    # pip install romanisim==0.1
    # to be able to run the tests for a specific version, use instead
    # pip install romanisim[test]==0.1

    # get stpsf data and untar it
    mkdir -p $HOME/data/stpsf-data
    cd $HOME/data/stpsf-data
    wget PATH_TO_STPSF_FILES -O stpsf-data.tar.gz
    tar -xzf stpsf-data.tar.gz
    export STPSF_PATH=$PWD/stpsf-data

    # get galsim galaxy catalogs
    # Note: ~5 GB each, takes a little while to download.
    # Both are needed for tests.  Neither are needed if you are
    # exclusively using analytic model galaxies.
    galsim_download_cosmos -s 23.5
    galsim_download_cosmos -s 25.2

The path to the STPSF data files may be found in their `documentation <https://stpsf.readthedocs.io/en/latest/installation.html>`_.

You may wish to, for example, set up a new python virtual environment
before running the above, or choose a different directory for
STPSF's data files.
