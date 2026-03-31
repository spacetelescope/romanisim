# romanisim Data Directory

This directory contains packaged reference data used by `romanisim` for simulations of the Roman Wide Field Instrument (WFI). The files included here provide static instrument inputs and auxiliary data required for image simulation, testing, and validation.

Bundling these data within the repository ensures reproducibility and minimizes reliance on external data access during runtime.

## Contents

The directory includes the following categories of data:

<!-- - **Instrument throughput / effective area tables**  
  Per-SCA filter throughput and effective-area tables stored as FITS files, along with the directory  
  `Roman_effarea_tables_20240327/`. -->

- **Technical reference data**  
  A local copy of the `roman-technical-information` repository, used to provide instrument parameters and reference values.

- **Auxiliary catalogs**  
  External data products used for simulation inputs (e.g.,  
  `COSMOS2020_CLASSIC_R1_v2.2_p3_Streamlined.fits`).

- **PSF-related data products**  
  Files used for point spread function modeling, including pupil masks and Zernike coefficient tables.

## Provenance and Usage

PSF-related data in this directory are derived from external Roman instrument modeling efforts:

- Zernike aberration data for Roman WFI were originally provided via **WebbPSF** data products.
- These data were processed into per-SCA formats using utilities developed within the GalSim repository (see `devel/external/parse_roman_zernikes_1217.py`).
- The processed data are currently distributed via GalSim and vendored here for use within `romanisim`.

The simulation code interpolates aberrations across the focal plane and applies wavelength scaling consistent with Roman optical models. Detailed descriptions of the implementation and assumptions are provided in the docstrings of the relevant PSF routines `romanisim.psf` and `romanisim.models.psf_utils`.

## Future Updates

Updated Roman instrument models are expected to supersede the currently bundled data:

- Zernike and pupil data from **Cycle 10** are anticipated for future use.
<!-- - These data products may require preprocessing to match the formats expected by existing simulation code. -->
<!-- - Migration of preprocessing utilities into `romanisim` is under consideration to improve transparency and reproducibility. -->

Until such updates are validated, the current data products remain the default.

## Data Management and Maintenance

- Some files in this directory duplicate external reference data (e.g., GalSim, WebbPSF, STPSF) to ensure stable and reproducible simulations.
- Updates to these files should be performed with care, as changes may impact PSF modeling, detector response, and downstream scientific results.
- Where feasible, upstream sources should be referenced and documented, and any transformations applied to the data should be reproducible and tracked.

## References

- WebbPSF documentation: https://webbpsf.readthedocs.io/
- GalSim repository: https://github.com/GalSim-developers/GalSim
- Roman Technical Information: https://github.com/spacetelescope/roman-technical-information