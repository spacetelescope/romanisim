Parameters
==========

The parameters module contains useful constants describing the Roman
telescope.  These include:

* the default read noise when CRDS is not used
* the number of border reference pixels (4)
* the specification of the read indices going into resultants for a
  fiducial L1 image for a handful of MA tables
* the read time (3.04 s)
* the default saturation limit
* the default IPC kernel
* the definition of the reference V2/V3 location in the focal plane to
  which to place the given right ascenscon and declination
* the default persistence parameters
* the default cosmic ray parameters
* the default gain when CRDS is not used

These values can be overridden by specifying a yaml config file on the
command line to romanisim-make-image.

See the `code
<https://github.com/spacetelescope/romanisim/blob/main/romanisim/parameters.py>`_
for the default values of these quantities.

.. automodapi:: romanisim.parameters

