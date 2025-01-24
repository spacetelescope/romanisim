Catalogs
========

The simulator takes catalogs describing objects in a scene and generates images of that scene.  These catalogs have the following form::

       ra     dec   type    n    half_light_radius    pa      ba     F087  
    float64 float64 str3 float64      float64      float64 float64 float64 
    ------- ------- ---- ------- ----------------- ------- ------- --------
      269.9    66.0  SER     1.6               0.6   165.6     0.9 1.80e-09
      270.1    66.0  SER     3.6               0.4    71.5     0.7 3.35e-09
      269.8    66.0  PSF    -1.0               0.0     0.0     1.0 2.97e-10
      269.9    66.0  SER     2.5               0.8   308.8     0.7 1.50e-09
      269.8    65.9  SER     3.9               0.9   210.0     0.9 3.28e-10
      270.1    66.0  SER     4.0               1.1   225.1     1.0 1.61e-09
      269.9    65.9  SER     1.5               0.3   271.8     0.6 1.13e-09
      269.9    65.9  SER     2.9               2.3    27.6     1.0 3.28e-09
      269.9    66.0  SER     1.1               0.3     4.3     1.0 9.99e-10


The following fields must be specified for each source:

* ra: the right ascension of the source
* dec: the declination of the source
* type: PSF or SER; whether the source is a point source or Sersic galaxy
* n: the Sersic index of the source.  This value is ignored if the source is a point source.
* half_light_radius: the half light radius of the source in arcseconds.  This value is ignored if the source is a point source.
* pa: the position angle of the source, in degrees east of north.  This value is ignored if the source is a point source.
* ba: the minor-to-major axis ratio.  This value is ignored if the source is a point source.

Following these required fields is a series of columns giving the fluxes of the the sources in "maggies"; the AB magnitude of the source is given by :math:`-2.5*\log_{10}(\mathrm{flux})`.  In order to simulate a scene in a given bandpass, a column with the name of that bandpass must be present giving the total fluxes of the sources.  Many flux columns may be present, and other columns may also be present but will be ignored.

The simulator then renders these images in the scene and produces the simulated L1 or L2 images.

The simulator API includes a few simple tools to generate parametric distributions of stars and galaxies.  The ``make_stars`` and ``make_galaxies`` routines make random catalogs of stars and galaxies.  The number of stars and galaxies can be adjusted.  Likewise, the power law index by which the sources' magnitudes are sampled can be adjusted, as can their limiting magnitudes.  Galaxy Sersic parameters, half-light radii, and position angles are chosen at random, with a rough attempt to make brighter galaxies appropriately larger (i.e., conserving surface brightness).  Stars can be chosen to be distributed with a King profile.  This functionality is however very rudimentary and limited, and is better suited for toy problems than real scientific work.  We expect scientific uses to be driven by custom-created catalogs rather than these simple routines.

.. automodapi:: romanisim.catalog
