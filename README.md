# README

This repository contains the code I use for the electromagnetic dosimetry research within my PhD.
The code for each published paper (conference or journal) is open sourced and freely available in `playground` directory.
To reproduce the results, `dosipy` should be installed by pulling the contents of this repository on your local machine and running
```shell
pip install .
```
The best practice is to create a local environment, e.g., using `conda`,
```shell
conda create --name dosipy python=3.9
```
and, inside the environment, installing all dependencies from `requirements.txt`
```shell
pip install -r requirements.txt
```

## Contents

| Directory | Subdirectory/Contents | Description |
|:---:|:---:|:---:|
| `dosipy` |  | Simple Python package for high-frequency EM & thermal dosimetry simulation and analysis. |
| 1 | `data` | Various source & target EM models. |
| 2 | `utils` | Data-loading, integration, differentiation & visualization. |
| 3 | bhte.py | 3-D bio-heat equation solver based on pseudo-spectral time domain approach. |
| 4 | constants.py | EM constants. |
| 5 | field.py | Assessment of electromagnetic field in free space radiated by half-wave dipole. |
| `playground` |  | Each directory within holds the code (notebooks and scripts) for journal and  conference papers, talks, and demos used throughout my research dealing  with high-frequency EM dosimetry. |
| 1 | `ACROSS2021_presentation` [published] | Accurate numerical approach to solving the surface integral of a vector field Presented at the 6-th Int'l Workshop on Advanced Cooperative Systems on December 3rd, 2021 |
| 2 | `BioEM2022_paper` [published] | Novel procedure for spatial averaging of absorbed power density on realistic body models at millimeter waves In proceedings of BioEM 2022, 2022, p. 242-248. |
| 3 | `IEEE-J-ERM_paper` [WIP] | Area-Averaged Transmitted and Absorbed Power Density on Realistic Body Parts Submitted to IEEE Journal of Electromagnetics, RF and Microwaves in Medicine and Biology. |
| 4 | `IEEE-TEMC_paper` [published] | Assessment of incident power density on spherical head model up to 100 GHz In IEEE Transactions on Electromagnetic Compatibility, 2022, doi: 10.1109/TEMC.2022.3183071 |
| 5 | `IMBioC2022_paper` [published] | Assessment of area-average absorbed power density on realistic tissue models at mmWaves In proceedings of 2022 IEEE MTT-S International Microwave Biomedical Conference (IMBioC), p. 153-155, doi: 10.1109/IMBioC52515.2022.9790150 |
| 6 | `IRPA2022_paper` [WIP] | Machine learning-assisted antenna modeling for realistic assessment of human exposure reference levels above 6 GHz  Abstract is presented at the 6th European Congress on Radiation Protection. Full paper is currently being prepared. |
| 7 | `SoftCOM2022_paper` [pending publishing] | Stochastic-Deterministic Electromagnetic Modeling of Human Head Exposure to Microsoft HoloLens To be in proceedings of the 30th International Conference on Software, Telecommunications and Computer Networks, 2022. |
| 8 | `demos` | Set of notebooks that showcase how to use `dosipy` package. |

 ## License

 [MIT](https://github.com/antelk/EMF-exposure-analysis/blob/main/LICENSE)