# Roman/WFI Imaging Effective Areas

## Included files

| Filename| Description|
|---------|------------|
| Roman_zeropoints_20240301.ecsv | Roman/WFI imaging filter zero points by SCA |

## Description

The following is included in the metadata of the ECSV file and may be retrieved as:

```
from astropy.table import Table

table = Table.read('Roman_zeropoints_20240301.ecsv')
print(table.meta['comments'])
```

Zero points for each detector and imaging optical element (filter) using effective area curves as of 2024 03 01.
The zeropoints are computed using synphot version 1.4.0. The method unit_response() 
determines the flux density that generates a count rate of 1 count per second through 
the bandpass. This is computed as:

unit response = $\dfrac{hc}{\int P_\lambda \lambda d\lambda}$

where h is the Planck constant, c is the speed of light, P is the effective 
area, and lambda is the wavelength. The integrals are approximated using the trapezoid 
method. Conversions to other units are performed at the pivot wavelength, which is a 
measure of the effective wavelength of the bandpass. The pivot wavelength is defined as:

pivot wavelength = $\sqrt{\dfrac{\int P_\lambda \lambda d\lambda}{\int (P_\lambda / \lambda) d\lambda}}$