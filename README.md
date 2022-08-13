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

| Directory | Contents |
| :-------- | :------- |
| `dosipy` | Simple Python package for high-frequency EM dosimetry simulation and analysis |
| `playground` | Each directory holds the code (notebooks and scripts) for journal and conference papers, talks, and demos used throughout my research dealing with high-frequency EM dosimetry |

 ## License

 [MIT](https://github.com/antelk/EMF-exposure-analysis/blob/main/LICENSE)