Image Input
===========

The simulator can alternatively take as input a catalog that references a separate list of images of sources.  The format of the catalog is described below, and the underlying "list of images" is described by a ``RealGalaxyCatalog``; see :ref:`Constructing a RealGalaxyCatalog`.  The concept here is that users may have noiseless images of galaxies from, for example, hydro simulations, and may wish to inject those into simulated images at specific locations.  Or users may have reference images from, for example, the COSMOS field, which could be used as truth images for simulations.

The simulator leverages GalSim's ``RealGalaxyCatalog`` framework to support this use case.  There are a handful of existing catalogs of images in the appropriate format for use with the ``RealGalaxyCatalog``---in particular, `GalSim's COSMOS reference catalog <https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog>`_ .

Using images as input comes with some subtleties.  The input images must have some well-defined PSF, which is specified as part of the reference catalog.  When these images are rendered in the scene, the input PSF is deconvolved before convolution with the Roman PSF.  As long as the input PSF is sharper than the output PSF, this causes no issues, but when the input PSF is larger than the output PSF, this requires deconvolving the input images, which usually produces ringing artifacts.  Care must be taken to provide input images with sharper PSFs than required in the output images to avoid this kind of artifact.

For image input, the simulator takes a catalog with the following structure::

     ra     dec   ident  rotate shear_pa shear_ba  dilate   F062  
  float64 float64 int64 float64 float64  float64  float64 float64 
  ------- ------- ----- ------- -------- -------- ------- ------- 
   269.9    66.0     0    303.2    24.2     0.9      0.3  4.9e-08 
   270.0    66.0     1    220.8   307.5     0.6      0.1  3.8e-07 
   269.9    66.1     0    194.0   134.0     0.7      0.4  3.6e-07 
   269.8    66.0     1    356.9   244.6     0.3      0.1  9.4e-08 
   269.9    66.1     0     91.0   335.1     0.9      0.1  5.0e-06 
   269.8    66.0     1     26.6   167.0     0.6      0.4  5.0e-08 
   269.9    66.0     0    152.3   103.7     0.5      0.3  3.4e-07 
   269.9    66.1     1    233.8    53.4     0.6      0.3  1.9e-06 
   269.8    66.0     0     97.8   323.9     0.6      0.2  3.1e-06 
   269.8    66.1     1     83.4   343.1     0.9      0.1  2.9e-08 
   269.9    66.0     0    219.1   347.5     0.7      0.0  3.2e-07 
   269.9    66.1     1     6.91   109.7     0.5      0.4  2.5e-07 
   269.9    66.0     0    12.04    44.2     0.6      0.1  2.7e-07

Moreover, the table metadata must include the keyword ``real_galaxy_catalog_filename`` specifying the location of the ``RealGalaxyCatalog`` catalog file.  The ``ident`` keyword then specifies the id of the galaxy image to use from the RealGalaxyCatalog, which is rendered at the specified location (ra, dec) and is optionally rotated (rotate), sheared (shear_pa, shear_ba), and dilated (dilate).  Finally, total fluxes in maggies (see the catalog :doc:`docs </romanisim/catalog>`) in specified bandpasses are given.

The following fields must be specified for each source:

* ra: the right ascension of the source
* dec: the declination of the source
* ident: the ID of the source image in the RealGalaxyCatalog
* rotate: the amount to rotate the source relative to its orientation in the RealGalaxyCatalog (degrees)
* shear_pa: the angle on which to shear the galaxy (i.e., beta `here <https://galsim-developers.github.io/GalSim/_build/html/shear.html#the-shear-class>`_) (degrees)
* shear_ba: the minor-to-major axis ratio of the shear (i.e., q `here <https://galsim-developers.github.io/GalSim/_build/html/shear.html#the-shear-class>`_)
* dilate: the amount to dilate (>1) or shrink (<1) the source

Following these required fields is a series of columns giving the fluxes of the the sources in "maggies", as in the catalog :doc:`docs </romanisim/catalog>`.

The simulator then renders these images in the scene and produces the simulated L1 or L2 images.


Constructing a RealGalaxyCatalog
================================

The simulator contains a routine intended to make it easier to construct a ``RealGalaxyCatalog`` from lists of input images and their associated PSF.  This helper routine does not expose all of the functionality of the ``RealGalaxyCatalog``, but it can at least help get one started.  Given a list of fits files containing images of galaxies and an image of their PSF, the routine builds the index file, image file, and PSF file needed by the image input mode.  One can then specify the index file name in the ``real_galaxy_catalog_filename`` attribute of the catalog and generate sources.

See the ``test_image_input`` `unit test <https://github.com/spacetelescope/romanisim/blob/main/romanisim/tests/test_image.py>`_ for a fully worked example.

