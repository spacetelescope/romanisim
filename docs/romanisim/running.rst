Running the simulation
======================

The primary means by which we expect most users to make images is
the command line interface::

    romanisim-make-image out.asdf

The combination of ``romanisim-make-image`` and various user-generated
input catalogs allows most of the simulator functionality to be exercised [#chromatic]_.

.. highlight:: none

The ``romanisim-make-image`` command line interface (CLI) has a number of arguments to support
this functionality::

    romanisim-make-image -h
    usage: romanisim-make-image [-h] [--bandpass BANDPASS] [--boresight] [--catalog CATALOG] [--config CONFIG] [--date DATE] [--level LEVEL] [--ma_table_number MA_TABLE_NUMBER] [--nobj NOBJ] [--previous PREVIOUS] [--radec RADEC RADEC] [--rng_seed RNG_SEED] [--roll ROLL] [--sca SCA] [--usecrds] [--stpsf] [--truncate TRUNCATE] [--pretend-spectral PRETEND_SPECTRAL] [--drop-extra-dq] [--scale-factor SCALE_FACTOR] [--extra-counts EXTRA_COUNTS [EXTRA_COUNTS ...]] filename

    Make a demo image.

    positional arguments:
      filename              output image (asdf)

    options:
      -h, --help            show this help message and exit
      --bandpass BANDPASS   bandpass to simulate (default: F087)
      --boresight           radec specifies location of boresight, not center of WFI. (default: False)
      --catalog CATALOG     input catalog (ecsv) (default: None)
      --config CONFIG       input parameter override file (yaml) (default: None)
      --date DATE           UTC Date and Time of observation to simulate in ISOT format. (default: None)
      --level LEVEL         1 or 2, for L1 or L2 output (default: 2)
      --ma_table_number MA_TABLE_NUMBER
      --nobj NOBJ
      --previous PREVIOUS   previous simulated file in chronological order used for persistence modeling. (default: None)
      --radec RADEC RADEC   ra and dec (deg) (default: None)
      --rng_seed RNG_SEED
      --roll ROLL           Position angle (North towards YIdl) measured at the V2Ref/V3Ref of the aperture used. (default: 0)
      --sca SCA             SCA to simulate. Use -1 to generate images for all SCAs; include {} in filename for this mode to indicate where the SCA number should be filled, e.g.
                            l1_wfi{}.asdf (default: 7)
      --usecrds             Use CRDS for distortion map (default: False)
      --stpsf               Use stpsf for PSF (default: False)
      --truncate TRUNCATE   If set, truncate the MA table at given number of resultants. (default: None)
      --pretend-spectral PRETEND_SPECTRAL
                            Pretend the image is spectral. exposure.type and instrument.element are updated to be grism / prism. (default: None)
      --drop-extra-dq       Do not store the optional simulated dq array. (default: False)
      --scale-factor SCALE_FACTOR
                            Velocity aberration-induced scale factor. If negative, use given time to calculated based on orbit ephemeris. (default: -1.0)
      --extra-counts EXTRA_COUNTS [EXTRA_COUNTS ...]
                        An optional FITS file to read to get an array of counts to add into the simulated image.Useful for wrapping
                        idealized images.If 2 arguments are sent in, then the second argument is assumed to be the HDU to use
                        (default=0) (default: None)

    EXAMPLE: romanisim-make-image output_image.asdf

Expected arguments controlling things like the input :doc:`here </romanisim/catalog>` to
simulate, the right ascension and declination of the telescope
[#boresight]_, the :doc:`bandpass </romanisim/bandpass>`, the Sensor
Chip Assembly (SCA) to
simulate, the level of the image to simulate (:doc:`L1 </romanisim/l1>`
or :doc:`L2 </romanisim/l2>`), the MA table to use, and the time of
the observation.

Additional arguments control some details of the simulation.  The
``--usecrds`` argument indicates that reference files should be pulled
from the Roman CRDS server, and is recommended.  The ``--stpsf``
argument indicates that the `STPSF
<https://stpsf.readthedocs.io>`_ package should be used to simulate
the PSF; note that this presently disables chromatic PSF rendering.

The ``--rng_seed`` argument specifies a seed to the random number
generator, enabling reproducible results.

The ``--nobj`` argument is only used when a catalog is not specified,
and controls the number of objects that are simulated in that case.

The ``previous`` argument specifies the previous simulated frame.
This information is used to support :doc:`persistence </romanisim/l1>`
modeling.

The ``--extra-counts`` argument(s) allows the user to pass in a FITS file with an 
array of counts (not counts/time). If a second argument is passed, it is assumed to be the HDU 
number to read.
This argument is useful to wrap idealized images into the Roman L1/L2 datamodel, 
including detector effects. You will probably want to set ``--nobj 0`` here to avoid 
simulating additional sources. We do not get any additional information/metadata from this 
file, so you will need to set other parameters (e.g. ``--bandpass``, ``--date``, ``--radec``, etc)
appropriately to match the image you are wrapping. 

.. [#chromatic] An important exception is the chromatic PSF rendering and 
   photon-shooting modes of GalSim; the current catalog format does 
   not support chromatic PSF rendering, and just assumes that all 
   sources are "gray" within a bandpass. 

.. [#boresight] This right ascension corresponds to either the
		location of the center of the WFI array or the
		telescope boresight, when the ``--boresight`` argument
		is specified.


Example
=======

Let's put the pieces together. Below, we create a synthetic catalog of bright
stars centered on the center of M13, observed with SCA 1 in the F087 filter.
Though we create 10,000 stars, only 86 will ultimately fall on SCA 1.
In this Python example, we synthesize images of those stars with ``romanisim`` via
the `~romanisim.image.simulate` function, and plot the results
with ``matplotlib``.

.. code-block:: python

    from copy import deepcopy

    import asdf
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord
    from astropy.visualization import simple_norm
    from galsim import UniformDeviate

    from romanisim import persistence, wcs
    from romanisim.catalog import make_stars
    from romanisim.image import simulate
    from romanisim.parameters import default_parameters_dictionary

    # use the coordinate of M13 as the center of the detector:
    coord = SkyCoord.from_name("M13")

    # choose an SCA to simulate:
    sca = 1
    filt = 'F087'
    seed = 0

    # save the star catalog here:
    catalog_path = 'small-synthetic-catalog.ecsv'

    # write out the result to ASDF:
    output_path = 'small-synthetic-image.asdf'

    # generate a stellar catalog:
    cat = make_stars(
        coord=coord,
        n=10_000,
        radius=0.7,
        bandpasses=[filt],
        faintmag=18,
        rng=UniformDeviate(seed=seed)
    )
    cat.write(catalog_path, overwrite=True)

    # prepare inputs for the `romanisim.image.simulate` method:
    metadata = deepcopy(default_parameters_dictionary)
    metadata['instrument']['detector'] = f'WFI{sca:02d}'
    metadata['instrument']['optical_element'] = filt
    metadata['exposure']['ma_table_number'] = 1
    wcs.fill_in_parameters(
        metadata, coord, boresight=False
    )

    # run the simulation:
    im, simcatobj = simulate(
        metadata, cat, webbpsf=True, level=2,
        persistence=persistence.Persistence(),
        rng=UniformDeviate(seed), usecrds=False
    )

    asdf_file = asdf.AsdfFile()
    romanisimdict = {'simcatobj': simcatobj}
    asdf_file.tree = {'roman': im, 'romanisim': romanisimdict}

    # plot a portion of the resulting rate image:
    fig, ax = plt.subplots()
    image = np.array(asdf_file.tree['roman']['data'])
    norm = simple_norm(image, 'asinh', asinh_a=1e-4)
    ax.imshow(image, norm=norm)

    ax.set(
        xlim=[2900, 3150],
        ylim=[3300, 3550]
    )

