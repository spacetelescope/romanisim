Running the simulation
======================

The primary means by which we expect most users to make images is
the command line interface::

    romanisim-make-image out.asdf

The combination of ``romanisim-make-image`` and various user-generated
input catalogs allows most simulator functionality to be exercised [#chromatic]_.

.. highlight:: none

The ``romanisim-make-image`` CLI has a number of arguments to support
this functionality::

    romanisim-make-image -h
    usage: romanisim-make-image [-h] [--catalog CATALOG] [--radec RADEC RADEC] [--bandpass BANDPASS]
                                [--sca SCA] [--usecrds] [--webbpsf] [--date DATE [DATE ...]]
                                [--level LEVEL] [--ma_table_number MA_TABLE_NUMBER] [--seed SEED]
                                [--nobj NOBJ] [--boresight] [--previous PREVIOUS]
                                filename
    
    Make a demo image.
    
    positional arguments:
      filename              output image (fits)
    
    optional arguments:
      -h, --help            show this help message and exit
      --catalog CATALOG     input catalog (csv) (default: None)
      --radec RADEC RADEC   ra and dec (deg) (default: None)
      --bandpass BANDPASS   bandpass to simulate (default: F087)
      --sca SCA             SCA to simulate (default: 7)
      --usecrds             Use CRDS for distortion map (default: False)
      --webbpsf             Use webbpsf for PSF (default: False)
      --date DATE [DATE ...]
                            Date of observation to simulate: year month day hour minute second
                            microsecond (default: None)
      --level LEVEL         1 or 2, for L1 or L2 output (default: 2)
      --ma_table_number MA_TABLE_NUMBER
      --seed SEED
      --nobj NOBJ
      --boresight           radec specifies location of boresight, not center of WFI. (default: False)
      --previous PREVIOUS   previous simulated file in chronological order used for persistence modeling.
                            (default: None)
    
    EXAMPLE: romanisim-make-image output_image.fits

Expected arguments controlling things like the input :doc:`here </romanisim/catalog>` to
simulate, the right ascension and declination of the telescope
[#boresight]_, the :doc:`bandpass </romanisim/bandpass>`, the SCA to
simulate, the level of the image to simulate (:doc:`L1 </romanisim/l1>`
or :doc:`L2 </romanisim/l2>`), the MA table to use, and the time of
the observation.

Additional arguments control some details of the simulation.  The
``--usecrds`` argument indicates that reference files should be pulled
from the Roman CRDS server; this is the recommended option when CRDS
is available.  The ``--webbpsf`` argument indicates that the `WebbPSF
<https://webbpsf.readthedocs.io>`_ package should be used to simulate
the PSF; note that this presently disables chromatic PSF rendering.

The ``--seed`` argument specifies a seed to the random number
generator, enabling reproducible results.

The ``--nobj`` argument is only used when a catalog is not specified,
and controls the number of objects that are simulated in that case.

The ``previous`` argument specifies the previous simulated frame.
This information is used to support :doc:`persistence </romanisim/l1>`
modeling.

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

