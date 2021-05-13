import numpy as np
import pyfftw
from scipy.integrate import odeint
from scipy.special import erfc

from .constants import pi


def init_temp(z, k, rho, C, rho_b, C_b, m_b, h_0, T_a, T_c, T_f, Q_m):
    """Return the temperature distribution by solving 1-D bioheat
    equation analytically over tissue depth. This can serve as the
    initial temperature distribution before solving bioheat equation
    numerically via the pseudo-spectral method.

    Ref: Deng, ZH; Liu, J. Analytical study on bioheat transfer
    problems with spatial or transient heating on skin surface or
    inside biological bodies, J Biomech Eng. Dec 2002, 124(6): 638-649,
    DOI: 10.1115/1.1516810

    Parameters
    ----------
    z : numpy.ndarray
        one dimensional solution domain
    k : float
        thermal conductivity of the tissue
    rho : float
        tissue density
    C : float
        heat capacity of the tissue
    rho_b : float
        blood density
    C_b : float
        blood heat capacity
    m_b : float
        blood perfusion
    h_0 : float
        heat convection coefficient between the skin surface and air
    T_a : float
        arterial temperature
    T_c : float
        body core temperature
    T_f : float
        surrounding air temperature
    Q_m : float
        metabolic heat generation

    Returns
    -------
    numpy.ndarray
        initial temperature distribution prior to radiation
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


def delta_temp_analytic(t, pen_depth, k, rho, C, I0, T_tr):
    """Return the closed-form solution of the 1-D BHTE with no blood
    perfusion considered over given simulation period, `t`.

    Ref: Foster, KR; Ziskin, MC; Balzano, Q. Thermal response of human
    skin to microwave energy: A critical review. Health Phys. Dec 2002,
    111(6): 528-541, DOI: 10.1097/HP.0000000000000571

    Parameters
    ----------
    sim_time : numpy.ndarray
        Simulation time
    pen_depth : float
        energy penetration depth
    k : float
        thermal conductivity of the tissue
    rho : float
        tissue density
    C : float
        heat capacity of the tissue
    I0 : float
        incident power density of the tissue surface
    T_tr : float
        transmission coefficient into the tisse

    Returns
    -------
    numpy.ndarray
        rise in temperature over exposure time
    """
    C_1 = 2 * I0 * T_tr / np.sqrt(pi * k * rho * C)
    C_2 = I0 * T_tr * pen_depth / k
    tau = 4 / pi * (C_2 / C_1) ** 2
    return (C_1 * np.sqrt(t)
            - C_2 * (1 - np.exp(t / tau) * erfc(np.sqrt(t / tau))))


def delta_temp(t, N, pen_depth, k, rho, C, m_b, I0, T_tr):
    """Numerical solution to 1-D Pennes' bioheat transfer equation by
    using Fast Fourier Transform on spatial coordinate.

    Parameters
    ----------
    t : numpy.ndarray
        simulation time; exposure time in seconds
    N : int
        number of collocation points
    pen_depth : float
        energy penetration depth
    k : float
        thermal conductivity of the tissue
    rho : float
        tissue density
    C : float
        heat capacity of the tissue
    m_b : float
        blood perfusion
    I0 : float
        incident power density of the tissue surface
    T_tr : float
        transmission coefficient into the tisse

    Returns
    -------
    numpy.ndarray
        temperature distribution in time for each collocation point
    """
    dx = pen_depth / N
    x = np.linspace(0, pen_depth, N)
    kappa = 2 * pi * np.fft.fftfreq(N, d=dx)
    SAR = I0 * T_tr / (rho * pen_depth) * np.exp(-x / pen_depth)
    SAR_fft = np.fft.fft(SAR)

    # initial conditions -- prior to radiofrequency exposure
    T0 = np.zeros_like(x)
    T0_fft = np.fft.fft(T0)

    # recasting complex numbers to an array for easier handling in SciPy
    T0_fft_ri = np.concatenate((T0_fft.real, T0_fft.imag))

    def rhs(T_fft_ri, t, kappa, k, rho, C, m_b, SAR_fft):
        T_fft = T_fft_ri[:N] + (1j) * T_fft_ri[N:]
        d_T_fft = (
            - np.power(kappa, 2) * k * T_fft / (rho * C)
            - rho * m_b * T_fft
            + SAR_fft / C
            )
        return np.concatenate((d_T_fft.real, d_T_fft.imag)).astype(np.float64)

    T_fft_ri = odeint(rhs, T0_fft_ri, t, args=(kappa, k, rho, C, m_b, SAR_fft))
    T_fft = T_fft_ri[:, :N] + (1j) * T_fft_ri[:, N:]

    deltaT = np.empty_like(T_fft)
    for i in range(t.size):
        deltaT[i, :] = np.fft.ifft(T_fft[i, :])
    return deltaT.real


def temp3(t, N, area, pen_depth, k, rho, C, rho_b, C_b, m_b, h_0, T_a, T_c,
          T_f, Q_m, SAR):
    """Numerical solution to 1-D Pennes' bioheat transfer equation by
    using Fast Fourier Transform on spatial coordinates.

    Parameters
    ----------
    t : numpy.ndarray
        simulation time; exposure time in seconds
    N : tuple
        collocation points in x, y and z direction
    area : tuple
        length and width of the heated surface area
    pen_depth : float
        energy penetration depth
    k : float
        thermal conductivity of the tissue
    rho : float
        tissue density
    C : float
        heat capacity of the tissue
    rho_b : float
        blood density
    C_b : float
        blood heat capacity
    m_b : float
        blood perfusion
    h_0 : float
        heat convection coefficient between the skin surface and air
    T_a : float
        arterial temperature
    T_c : float
        body core temperature
    T_f : float
        surrounding air temperature
    Q_m : float
        metabolic heat generation
    SAR : numpy.ndarray
        3-D array of shape (`N[0]`, `N[1]`, `N[2]`), each value
        corresponds to (x, y, z) SAR value

    Returns
    -------
    numpy.ndarray
        temperature distribution in time for each collocation point
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
    # lapinv = np.zeros_like(lap)
    # lapinv[lap != 0] = 1. / lap[lap != 0]
    # DX = 1j * KX * lapinv
    # DY = 1j * KY * lapinv
    # DZ = 1j * KZ * lapinv
    # lap = DX + DY + DZ
    
    T0 = np.ones(N) * init_temp(z, k, rho, C, rho_b, C_b, m_b, h_0, T_a, T_c,
                                T_f, Q_m)
    T0 = T0.ravel()
    
    
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

    
    T = odeint(rhs, T0, t)
    return T.reshape(-1, Nx, Ny, Nz)
