"""Bioheat transfer equation solvers."""

import numpy as np
from scipy.integrate import odeint
from scipy.special import erfc

from .constants import pi


def init_temp(z, k, rho_b, C_b, m_b, h_0, T_a, T_c, T_f, Q_m):
    """Return the temperature distribution by solving the 1-D bioheat
    equation analytically over tissue depth. This can serve as the
    initial temperature distribution before solving bioheat equation
    numerically via the pseudo-spectral method.

    Ref: Deng, ZH; Liu, J. Analytical study on bioheat transfer
    problems with spatial or transient heating on skin surface or
    inside biological bodies, J Biomech Eng. 2002, 124(6): 638-649,
    DOI: 10.1115/1.1516810

    Parameters
    ----------
    z : numpy.ndarray
        1-D space representing the solution domain.
    k : float
        Thermal conductivity of the tissue.
    rho_b : float
        Blood density.
    C_b : float
        Blood heat capacity.
    m_b : float
        Volumetric blood perfusion.
    h_0 : float
        Heat convection coefficient between a skin surface and the air.
    T_a : float
        Arterial temperature.
    T_c : float
        Body core temperature.
    T_f : float
        Surrounding air temperature.
    Q_m : float
        Metabolic heat generation.

    Returns
    -------
    numpy.ndarray
        Initial temperature distribution prior to exposure.
    """
    pen_depth = np.max(z)
    w_b = m_b * rho_b * C_b
    A = w_b / k
    denom = (np.sqrt(A) * np.cosh(np.sqrt(A) * pen_depth)
             + (h_0 / k) * np.sinh(np.sqrt(A) * pen_depth))
    numer = ((T_c - T_a - Q_m / w_b) * (np.sqrt(A) * np.cosh(np.sqrt(A) * z)
             + (h_0 / k) * np.sinh(np.sqrt(A) * z))
             + h_0 / k * (T_f - T_a - Q_m / w_b) * np.sinh(np.sqrt(A)
             * (pen_depth - z)))
    return T_a + Q_m / w_b + numer / denom


def delta_temp_analytic(t, pen_depth, k, rho, C, IPD, T_tr):
    """Return the closed-form solution of the 1-D bioheat equation with
    no blood perfusion considered over given simulation period, `t`.

    Ref: Foster, KR; Ziskin, MC; Balzano, Q. Thermal response of human
    skin to microwave energy: A critical review. Health Phys. 2002,
    111(6): 528-541, DOI: 10.1097/HP.0000000000000571

    Parameters
    ----------
    t : numpy.ndarray
        Simulation time.
    pen_depth : float
        Energy penetration depth.
    k : float
        Thermal conductivity of the tissue.
    rho : float
        Tissue density.
    C : float
        Heat capacity of the tissue.
    IPD : float
        Incident power density of the tissue surface
    T_tr : float
        Transmission coefficient into the tisse.

    Returns
    -------
    numpy.ndarray
        Temperature rise during the time of exposure.
    """
    C_1 = 2 * IPD * T_tr / np.sqrt(pi * k * rho * C)
    C_2 = IPD * T_tr * pen_depth / k
    tau = 4 / pi * (C_2 / C_1) ** 2
    return (C_1 * np.sqrt(t)
            - C_2 * (1 - np.exp(t / tau) * erfc(np.sqrt(t / tau))))


def delta_temp_1d(t, N, pen_depth, k, rho, C, m_b, IPD, T_tr):
    """Numerical solution of the 1-D bioheat equation by using the FFT
    on a spatial coordinate.

    Parameters
    ----------
    t : numpy.ndarray
        Simulation time.
    N : int
        Number of collocation points.
    pen_depth : float
        Energy penetration depth.
    k : float
        Thermal conductivity of the tissue-
    rho : float
        Tissue density.
    C : float
        Heat capacity of the tissue.
    m_b : float
        Volumetric blood perfusion.
    IPD : float
        Incident power density to the tissue surface.
    T_tr : float
        Transmission coefficient into the tisse.

    Returns
    -------
    numpy.ndarray
        Temperature distribution in time and space.
    """
    dx = pen_depth / N
    x = np.linspace(0, pen_depth, N)
    kappa = 2 * pi * np.fft.fftfreq(N, d=dx)
    SAR = IPD * T_tr / (rho * pen_depth) * np.exp(-x / pen_depth)
    SAR_fft = np.fft.fft(SAR)

    def rhs(T_fft_ri, t, kappa, k, rho, C, m_b, SAR_fft):
        T_fft = T_fft_ri[:N] + (1j) * T_fft_ri[N:]
        d_T_fft = (
            - np.power(kappa, 2) * k * T_fft / (rho * C)
            - rho * m_b * T_fft
            + SAR_fft / C
            )
        return np.concatenate((d_T_fft.real, d_T_fft.imag)).astype(np.float64)

    # initial conditions - prior to radiofrequency exposure
    T0 = np.zeros_like(x)
    T0_fft = np.fft.fft(T0)
    # recasting complex numbers to an array for easier handling in `scipy`
    T0_fft_ri = np.concatenate((T0_fft.real, T0_fft.imag))
    T_fft_ri = odeint(rhs, T0_fft_ri, t, args=(kappa, k, rho, C, m_b, SAR_fft))
    T_fft = T_fft_ri[:, :N] + (1j) * T_fft_ri[:, N:]
    deltaT = np.empty_like(T_fft)
    for i in range(t.size):
        deltaT[i, :] = np.fft.ifft(T_fft[i, :])
    return deltaT.real


def temp_3d(t, N, area, pen_depth, k, rho, C, rho_b, C_b, m_b, h_0, T_a, T_c,
            T_f, Q_m, SAR):
    """Numerical solution of the 1-D bioheat equation by using the FFT
    on spatial coordinates.

    Parameters
    ----------
    t : numpy.ndarray
        Simulation time.
    N : tuple
        Number of collocation points in x-, y- and z-direction.
    area : tuple
        Length and width of the exposure surface of a human tissue.
    pen_depth : float
        Energy penetration depth.
    k : float
        Thermal conductivity of a tissue.
    rho : float
        Tissue density.
    C : float
        Tissue heat capacity.
    rho_b : float
        Blood density.
    C_b : float
        Blood heat capacity.
    m_b : float
        Volumetric blood perfusion.
    h_0 : float
        Heat convection coefficient between the skin surface and air.
    T_a : float
        Arterial temperature.
    T_c : float
        Body core temperature.
    T_f : float
        Surrounding air temperature.
    Q_m : float
        Metabolic heat generation.
    SAR : numpy.ndarray
        3-D array of shape (`N[0]`, `N[1]`, `N[2]`) of specific
        absorption rate values.

    Returns
    -------
    numpy.ndarray
        Temperature distribution in time and space.
    """
    Nx, Ny, Nz = N
    X, Y = area
    Z = pen_depth
    dx = X / Nx
    dy = Y / Ny
    dz = Z / Nz
    z = np.linspace(0, Z, Nz)
    kx = 2 * pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2 * pi * np.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz)
    lap = KX ** 2 + KY ** 2 + KZ ** 2
    lapinv = np.zeros_like(lap)
    lapinv[lap != 0] = 1. / lap[lap != 0]
    DX = 1j * KX * lapinv
    DY = 1j * KY * lapinv
    DZ = 1j * KZ * lapinv
    lap = DX + DY + DZ

    def rhs(T, t):
        T = T.reshape(Nx, Ny, Nz)
        T_fft = np.fft.fftn(T, axes=(0, 1, 2))
        lapT_fft = - lap * T_fft
        lapT = np.fft.ifftn(lapT_fft, axes=(0, 1, 2))

        dTdt = (k * lapT
                + rho_b ** 2 * m_b * C_b * (T_a - T)
                + Q_m
                + SAR * rho) / (rho * C)
        return dTdt.real.ravel()

    _T0 = init_temp(z, k, rho_b, C_b, m_b, h_0, T_a, T_c, T_f, Q_m)
    T0 = np.ones((Nx, Ny, Nz)) * _T0
    T0 = T0.ravel()
    T = odeint(rhs, T0, t)
    return T.reshape(-1, Nx, Ny, Nz)
