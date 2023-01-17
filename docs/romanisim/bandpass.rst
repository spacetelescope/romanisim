Bandpasses
==========

The simulator can render scenes in a number of different bandpasses.  The choice of bandpass affects the point spread function used, the sky backgrounds, the fluxes of sources, and the reference files requested.

At present, romanisim simply passes the choice of bandpass to other packages---to webbpsf for PSF modeling, to galsim.roman for sky background estimation, to CRDS for reference file selection, or to the catalog for the selection of appropriate fluxes.  However, because catalog fluxes are specified in "maggies" (i.e., in linear fluxes on the AB scale), the simulator needs to know how to convert between a maggie and the number of photons Roman receives from a source.  Accordingly, the simulator knows about the AB zero points of the Roman filters, as derived from https://roman.gsfc.nasa.gov/science/WFI_technical.html .

One technical note: it is unclear what aperture is used for the bandpasses provided by Goddard.  The Roman PSF formally extends to infinity and some light is received by the detector but is so far from the center of the PSF that it is not useful for flux, position, or shape measurements.  Often for the purposes of computing effective area curves only light landing within a fixed aperture is counted.  Presently the simulator assumes that an infinite aperture is used.  This can result in an approximately 10% different flux scale than more reasonable aperture selections.

.. automodapi:: romanisim.bandpass
