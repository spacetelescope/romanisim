Bandpasses
==========

The simulator can render scenes in a number of different bandpasses.  The choice of bandpass affects the point spread function used, the sky backgrounds, the fluxes of sources, and the reference files requested.

At present, romanisim simply passes the choice of bandpass to other packages---to webbpsf for PSF modeling, to galsim.roman for sky background estimation, to CRDS for reference file selection, or to the catalog for the selection of appropriate fluxes.  However, because catalog fluxes are specified in "maggies" (i.e., in linear fluxes on the AB scale), the simulator needs to know how to convert between a maggie and the number of photons Roman receives from a source.  Accordingly, the simulator knows about the AB zero points of the Roman filters, as derived from https://roman.gsfc.nasa.gov/science/WFI_technical.html .
