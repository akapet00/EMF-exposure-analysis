# README

This repository contains the code I use for some of the electromagnetic (EM) dosimetry and exposure assessment research during my PhD.
The code for each published paper (conference or journal) is open sourced and freely available in `playground` directory.
To reproduce the results, `dosipy` should be installed by pulling the contents of this repository on your local machine and running
```shell
pip install .
```
The best practice is to create a local environment, e.g., using `conda`,
```shell
conda create --name dosipy python=3.9.12
```
and, inside the environment, installing all dependencies from `requirements.txt`
```shell
pip install -r requirements.txt
```

## Contents

| Directory | Subdirectory/Contents | Description |
|:---:|:---:|:---:|
| `dosipy` |  | Python package for high-frequency (⪆ 6 GHz, up to 300 GHz) EM & very simple thermal dosimetry simulation and analysis. |
| 1 | `data` | Various source & target EM models. |
| 2 | `utils` | Data-loading, integration, (automatic) differentiation & visualization. |
| 3 | bhte.py | 3-D bio-heat equation solver based on pseudo-spectral time domain approach. |
| 4 | constants.py | EM constants. |
| 5 | field.py | Assessment of EM field components in free space radiated by half-wave dipole with the help of automatic differentiation (JAX). |
| `playground` |  | Each subdirectory holds the code (notebooks and scripts) for the journal and conference papers, talks, and demos used throughout my research dealing with RF EM dosimetry. |
| 1 | `ACROSS2021_presentation` [published] | Accurate numerical approach to solving the surface integral of a vector field. Presented at the 2021 Int'l Workshop on Advanced Cooperative Systems.|
| 2 | `BioEM2022_paper` [published] | Novel procedure for spatial averaging of absorbed power density on realistic body models at millimeter waves. In proceedings of BioEM2022, p. 242-248.|
| 3 | `IEEE-J-ERM_paper` [published] | Area-averaged transmitted and absorbed power density on realistic body parts. In IEEE Journal of Electromagnetics, RF and Microwaves in Medicine and Biology, vol. 7, no.1, p. 39-45, 2022, doi: https://doi.org/10.1109/JERM.2022.3225380. |
| 4 | `IEEE-TEMC_paper` [published] | Assessment of incident power density on spherical head model up to 100 GHz. In IEEE Transactions on Electromagnetic Compatibility, 2022, doi: https://doi.org/10.1109/TEMC.2022.3183071 |
| 5 | `IMBioC2022_paper` [published] | Assessment of area-average absorbed power density on realistic tissue models at mmWaves. In proceedings of 2022 IEEE MTT-S International Microwave Biomedical Conference (IMBioC), p. 153-155, doi: 10.1109/IMBioC52515.2022.9790150 |
| 6 | `IRPA2022_paper` [published] | Machine learning-assisted antenna modeling for realistic assessment of incident power density on non-planar surfaces above 6 GHz. Abstract presented at 2022 European Congress on Radiation Protection. Full paper in Radiation Protection Dosimetry, 2023, doi: https://doi.org/10.1093/rpd/ncad114 |
| 7 | `SoftCOM2022_paper` [published] | Stochastic-deterministic electromagnetic modeling of human head exposure to Microsoft HoloLens. In proceedings of 2022 International Conference on Software, Telecommunications and Computer Networks (SoftCOM), p. 1-5, doi: https://doi.org/10.23919/SoftCOM55329.2022.9911431. |
| 8 | `SpliTech2021_paper` [published] | Application of automatic differentiation in electromagnetic dosimetry - assessment of the absorbed power density in the mmWave frequency spectrum. In proceedings of 2021 6th International Conference on Smart and Sustainable Technologies (SpliTech), p. 1-6, doi: https://doi.org/10.23919/SpliTech52315.2021.9566429. |
| 9 | `demos` | Self-contained notebooks showcasing how to properly use `dosipy`. |

 ## License

 [MIT](https://github.com/antelk/EMF-exposure-analysis/blob/main/LICENSE)