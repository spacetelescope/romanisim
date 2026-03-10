# Roman Space Telescope Technical Information

This repository hosts technical information related to the Nancy Grace Roman Space Telescope (Roman), its Wide Field Instrument (WFI), and its Coronagraph technology demonstration (Coronagraph). This repository is open to everyone, and the information contained within is public without any restrictions. The intended audience for this information is primarily the:
* Roman Science Operations Center (SOC) at the Space Telescope Science Institute (STScI)
* Roman Science Support Center (SSC) at the Infrared Processing and Analysis Center (IPAC)
* Project Infrastructure and Wide Field Science Teams
* Committees performing definition of the community survey programs
* Roman user community

# Using this Repository

The information in this respository is tagged for releases. We encourage users to use the released versions to better communicate the version of the information used, particularly as some items are updated frequently during the integration and test process.

Within the data/ directory, information related to the spacecraft, the WFI, and the Coronagraph are broken out into separate subdirectories. Each folder contains a README.md file that describes the information contained within as well as giving the update history for the information.

**Note:** Most files here are either in a .yaml or .ecsv format. Either can be easily read in with Python3.

## YAML Files 
YAML file can be read in with the [pyyaml packge](https://pyyaml.org). For more details, please see the [pyyaml documentation](https://pyyaml.org/wiki/PyYAMLDocumentation).
For example, here's how to read in the yaml file `MissionandObservatory.yaml` located in `roman-technical-information/data/Observatory/MissionandObservatoryTechnicalOverview/`:
```
# import the yaml package (installed via the pyyaml package)
import yaml

# read in the file
with open('MissionandObservatory.yaml', 'r') as file:
    # the data will be stored as the Python dictionary "Roman"
    Roman = yaml.safe_load(file)
    
# Roman is a Python dictionary
print(Roman)
print(Roman.keys())
print(Roman['Mission_and_Spacecraft_Parameters'])
print(Roman['Mission_and_Spacecraft_Parameters']['orbit'])
```

## ECSV (Enhanced Character-Separated Values) Files 
ECSV files can be read in with the [astropy package](https://www.astropy.org). For more details, please see [astropy's ecsv documentation](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) and the [astropy example on reading Gaia ecsv files](https://docs.astropy.org/en/stable/io/ascii/read.html#reading-gaia-data-tables).
For example, here's how to read in the ecsv file `nominal_roll_angles_dec_1_observatory.ecsv` located in `roman-technical-information/data/Observatory/RollAngles/`: 
```
# You can use either Table or QTable, both of which are part of the astropy package
from astropy.table import Table
from astropy.table import QTable

# Read in the ecsv file
Roman = Table.read("nominal_roll_angles_dec_1_observatory.ecsv",format="ascii.ecsv")

# "Roman" is now an astropy data table (https://docs.astropy.org/en/stable/table/)
type(Roman)
dir(Roman)
print(Roman.keys())
print(Roman['Month'])
print(Roman['RA_sun'].unit)

# You can alternatively use QTable instead
Roman = QTable.read("nominal_roll_angles_dec_1_observatory.ecsv",format="ascii.ecsv")
```

# Versioning
You can find the version number in VERSION.md.

The Roman Technical Information repo uses the following version number convention:

MAJOR.MINOR.PATCH

Where MAJOR is a major code change (such as the introduction of a many new parameters), MINOR is a minor code change (such as the introduction of a new table), and PATCH is a small patch/bugfix (e.g., fixing a typo). See [Semantic Versioning](https://semver.org) for more details.

# Contributions and Feedback

Please see our [contributing instructions](CONTRIBUTING.md) for more information.

Most importantly, if you are in need of information that is not listed here or that you suspect is out of date, please open an issue and someone will get back to you as soon as possible.

# Code of Conduct

This repository follows the Spacetelescope organization [Code of Conduct](CODE_OF_CONDUCT.md) to provide a welcoming community to all of our users and contributors.

# Documentation and Help

For more information related to the Roman mission, please see the following resources:
* [Roman mission website at Goddard Space Flight Center (GSFC)](https://roman.gsfc.nasa.gov/)
* [The Roman Documentation System (RDox) at STScI](https://roman-docs.stsci.edu/)

More links will be added as the become available.

For help with understanding or using the information in this repository, you can contact the [Roman Space Telescope Help Desk](https://stsci.service-now.com/roman).
