World Coordinate Systems & Distortion
=====================================

The simulator has two options for modeling distortion and world coordinate systems.  The first is to use the routines in the galsim.roman package; see GalSim's documentation for more information.  The second is to use distortion reference files from the Calibration References Data System (CRDS).

The latter system works by grabbing reference distortion maps for the appropriate detector and filter combinations from the Roman CRDS server.  These distortion maps are then wrapped into a galsim WCS object and fed to galsim's rendering pipeline.


.. automodapi:: romanisim.wcs
