# README

This repository contains the code I use for the electromagnetic (EM) dosimetry research within my PhD.
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
| `dosipy` |  | Python package for high-frequency (⪆ 6 GHz, up to 300 GHz) EM & simplistic thermal dosimetry simulation and analysis. |
| 1 | `data` | Various source & target EM models. |
| 2 | `utils` | Data-loading, integration, differentiation & visualization. |
| 3 | bhte.py | 3-D bio-heat equation solver based on pseudo-spectral time domain approach. |
| 4 | constants.py | EM constants. |
| 5 | field.py | Assessment of EM field components in free space radiated by half-wave dipole with the help of automatic differentiation (JAX). |
| `playground` |  | Each directory within holds the code (notebooks and scripts) for journal and  conference papers, talks, and demos used throughout my research dealing  with high-frequency EM dosimetry. |
| 1 | `ACROSS2021_presentation` [published] | Accurate numerical approach to solving the surface integral of a vector field. Presented at the 2021 Int'l Workshop on Advanced Cooperative Systems.|
| 2 | `BioEM2022_paper` [published] | Novel procedure for spatial averaging of absorbed power density on realistic body models at millimeter waves. In proceedings of BioEM2022, p. 242-248. |
| 3 | `IEEE-J-ERM_paper` [WIP] | Area-averaged transmitted and absorbed power density on realistic body parts. Submitted to IEEE Journal of Electromagnetics, RF and Microwaves in Medicine and Biology. |
| 4 | `IEEE-TEMC_paper` [published] | Assessment of incident power density on spherical head model up to 100 GHz. In IEEE Transactions on Electromagnetic Compatibility, 2022, doi: 10.1109/TEMC.2022.3183071 |
| 5 | `IMBioC2022_paper` [published] | Assessment of area-average absorbed power density on realistic tissue models at mmWaves. In proceedings of 2022 IEEE MTT-S International Microwave Biomedical Conference (IMBioC), p. 153-155, doi: 10.1109/IMBioC52515.2022.9790150 |
| 6 | `IRPA2022_paper` [WIP] | Machine learning-assisted antenna modeling for realistic assessment of incident power density on non-planar surfaces above 6 GHz. Abstract is presented at 2022 European Congress on Radiation Protection. Full paper is currently pending publication in Radiation Protection Dosimetry journal. |
| 7 | `SoftCOM2022_paper` [published] | Stochastic-Deterministic Electromagnetic Modeling of Human Head Exposure to Microsoft HoloLens. In proceedings of 2022 International Conference on Software, Telecommunications and Computer Networks (SoftCOM), p. 1-5, doi: 10.23919/SoftCOM55329.2022.9911431. |
| 8 | `demos` | Set of notebooks that showcase how to use `dosipy` package. |

 ## License

 [MIT](https://github.com/antelk/EMF-exposure-analysis/blob/main/LICENSE)