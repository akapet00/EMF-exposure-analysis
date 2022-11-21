# Thin wire modelling in frequency domain

This Matlab code is used for generating the spatial distribution of the complex current along the thin wire in free space.
Linear antenna approximation is defined via its length, L, and radius, a.
Spatial distribution of the current is obtained by solving the steady-state Pocklington integro-differential equation (`solver.m`) by using the indirect Galerkin-Bubnov boundary element method as described in

    Poljak, D. Advanced Modeling in Computational Electromagnetic Compatibility, 2006, Wiley, 10.1002/0470116889.

To generate `fs_current.mat` dataset, adjust the variables in `run_fs_current.mat` and run it in any version of Matlab newer than 2010a.