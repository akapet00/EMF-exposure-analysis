# Thin wire modelling in frequency domain

This code is used for generating the spatial distribution of the current along the thin wire in free space.
Linear antenna approximation is defined via its length, L, and radius, a.
Spatial distribution of the current is obtained by solving the steady-state Pocklington integro-differential equation as presribed in:

    Poljak, D. Advanced Modeling in Computational Electromagnetic Compatibility, 2006, Wiley, 10.1002/0470116889.

To generate `fs_current.mat` dataset, run the script `run_fs_current.mat` in Matlab (any version newer than 2010a will do).