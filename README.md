[![CI](https://github.com/spacetelescope/romanisim/actions/workflows/ci.yml/badge.svg)](https://github.com/spacetelescope/romanisim/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/spacetelescope/romanisim/branch/main/graph/badge.svg?token=pkoLtQOa2v)](https://codecov.io/gh/spacetelescope/romanisim)

# romanisim: an image simulator for Roman

romanisim is a Galsim-based simulator of imaging data from the Wide
Field Instrument (WFI) on the Nancy Grace Roman Space Telescope
(pronounced roman-eye-sim, stylized Roman I-Sim).  It uses
[Galsim](https://galsim-developers.github.io/GalSim/_build/html/overview.html)
to render astronomical scenes,
[WebbPSF](https://galsim-developers.github.io/GalSim/_build/html/overview.html)
to model the point spread function, and
[CRDS](https://github.com/spacetelescope/crds) to access the
calibration information needed to produce realistic WFI images.

The simulator starts by producing an idealized scene with Galsim and a
PSF, and then proceeds to simulate the various noise sources and
instrumental systematics imprinted by the system.  One major feature
is a fairly faithful implementation of up-the-ramp sampling and
ramp-fitting, so that romanisim can produce realistic L1 images ("raw" sets
of up the ramp samples like those that will be delivered from the telescope) and
L2 images (calibrated images of astronomical flux per pixel).

> **Warning**
> romanisim is under active developement.  Its output has not been formally validated; only limited testing has been performed.  For this reason, use of romanisim for preparation of ROSES proposals is not advised.  Other packages like galsim's roman package or STIPS may better serve such purposes.

## Documentation

See the full romanisim [documentation](https://romanisim.readthedocs.org) at readthedocs.

## Installation

    pip install romanisim

should do most of what you want.  Then

    romanisim-make-image out.asdf

will render a test image.  See the
[documentation](https://romanisim.readthedocs.org) for more
information about simulating scenes you're actually interested in!

## Contributing

romanisim is intended to support the community in understanding and
analyzing imaging from Roman.  If there are features you want to use or
see, file an
[issue](https://github.com/spacetelescope/romanisim/issues), or better
yet, make a pull request!
