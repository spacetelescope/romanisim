# Roman/WFI Focal Plane System (FPS) Sensor Chip Assembly (SCA) Performance Measurements

These measurements were performed on either individual flight SCAs in the NASA Goddard Detector Characterization Lab (DCL) at a temperature of 95 K or on the full flight focal plane during WFI Thermal Vacuum Test #2 (TVAC2) in the nominal operation (NomOp) test plateau at a temperature of 89.5 K. The test parameters for each performance measurement are noted in the descriptions. In these data tables individual detectors are described by either their Sensor Control Unit (SCU) number, which defines their position in the focal plane array as an integer 1 through 18, or their SCA serial number, which is a 5-digit integer that defines each individual detector. Please refer to the SCU to SCA mapping table and labeled diagram of the flight focal plane array for reference.

## Included files

| Filename| Description|
|---------|------------|
| WFI_SCU_to_SCA.ecsv | Roman/WFI SCU to SCA mapping. |
| WFI_detector_array_SCU_SCA_labeled.png | Roman/WFI detector array with labeled Sensor Control Unit (SCU) numbers and Sensor Chip Assembly (SCA) serial numbers. |
| WFI_CDS_Noise_summary.ecsv | Roman/WFI TVAC2 NomOp correlated double sampling (CDS) noise measurements (mean and medians) for each SCA. |
| WFI_Dark_current_summary.ecsv | Roman/WFI TVAC2 NomOp dark current measurements (means, medians, and percentage of pixels passing the dark current requirement) for each SCA. NOTE: These measurements are representative of the instrument internal thermal background rather than the true dark current floor of the detectors. |
| WFI_Persistence_summary.ecsv | Roman/WFI TVAC2 NomOp persistence measurements as a function of time for each SCA. The persistence decay was measured using 10 regularly spaced darks spanning at total time of 1770 secs (~30 min) following exposures of 56 frames each where the detectors were exposed to ~900 e-/s flat field illumination, for a total of ~159 ke- of charge accumulation. |
| WFI_Persistence_exp_fits.ecsv | Roman/WFI TVAC2 NomOp exponential decay fits to persistence measurements as a function of time for each SCA. The median persistence (e-/s over the dark current) was fit to an exponential function with this form: a*exp(-b*x)+d*exp(-e*x)+c, where x is the time in the decay curve and a, b, c, d, and e are fit coefficients provided in this file. |
| WFI_Pixels_meet_requirements.ecsv | Roman/WFI TVAC2 NomOp pixel operability statistics. For each SCA this includes the number of pixels deemed inoperable, which do not meet requirements ("bad pixels"), and the percentage of pixels deemed operable that do meet requirements ("good pixels") for each SCA. The performance metrics that can contribute to a pixel being inoperable are dark current, total noise, linearity, persistence, QE, flat field response, gain, and electrical connectivity. NOTE: SCU/SCA 4/21115 has a significantly lower operable pixel fraction due to an extended region of the detector with persistence that is somewhat larger than the requirement. |
| WFI_Quantum_efficiency.ecsv | Roman/WFI DCL measurements of the median quantum efficiency (QE) versus wavelength (linearly interpolated) for each SCA. |
| WFI_Total_noise.ecsv | Roman/WFI TVAC2 NomOp total noise measurements (means, medians, and percentage of pixels passing the total noise requirement) for each SCA. |
