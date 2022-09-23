APT file support
================

The simulator possesses rudimentary support for simulating images from APT files.  In order to simulate a scene, romanisim needs to know what's in the scene, as specified by a :doc:`catalog </romanisim/catalog>`.  It also needs to know where the telescope is pointed, the roll angle of the telescope, the date of the observation, and the bandpass.  Finally, it needs to know what the MultiAccum table of the observation is---roughly, how long the exposure is and how the reads of the detector should be averaged into resultants.

Much of this information is available in an APT file.  A rudimentary APT file reader can pull out the right ascension and declinations of observations, as well as the filters requested.  However, support for roll angles is not yet included.  APT files do not include the dates of observation, so this likewise is not included.  APT files naturally do not contain catalogs of sources in the field, so some provision must be made for adding this information.

This module is not yet fully baked.


