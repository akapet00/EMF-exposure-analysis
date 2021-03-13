import numpy as np
from scipy.integrate import odeint
from scipy.constants import (epsilon_0 as eps_0, mu_0, pi)
from scipy.special import erfc


def initT_depth_analytic(x, k, rho, C, rho_b, C_b, m_b, h_0, T_a, T_c, T_f, Q_m):
    r"""Return the temperature distribution by solving 1-D bioheat
    equation analytically over tissue depth. This can serve as the
    initial temperature distribution before solving bioheat equation
    numerically via the pseudo-spectral method.
    
    Ref: Deng, ZH; Liu, J. Analytical study on bioheat transfer
    problems with spatial or transient heating on skin surface or
    inside biological bodies, J Biomech Eng. Dec 2002, 124(6): 638-649,
    DOI: 10.1115/1.1516810
    
    Parameters
    ----------
    x : numpy.ndarray
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
        initial temperature distribution befor mW energy radiation
    """
    pen_depth = np.max(x)
    w_b = m_b * rho_b * C_b
    A = w_b / k
    denom = (
        np.sqrt(A) * np.cosh(np.sqrt(A) * pen_depth)
        + (h_0 / k) * np.sinh(np.sqrt(A) * pen_depth))
    numer = (
        (T_c - T_a - Q_m / w_b)
        * (np.sqrt(A) * np.cosh(np.sqrt(A) * x) + (h_0 / k) * np.sinh(np.sqrt(A) * x))
        + h_0 / k * (T_f - T_a - Q_m / w_b) * np.sinh(np.sqrt(A) * (pen_depth - x)))
    return T_a + Q_m / w_b + numer / denom


def deltaT_depth_analytic(t, pen_depth, k, rho, C, I0, T_tr):
    r"""Return the closed-form solution of the 1-D BHTE with no blood
    perfusion considered over given simulation period, t.
    
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
        rise in temperature over exposure time, t
    """
    C_1 = 2 * I0 * T_tr / np.sqrt(pi * k * rho * C)
    C_2 = I0 * T_tr * pen_depth / k
    tau = 4 / pi * (C_2 / C_1) ** 2
    return (
        C_1 * np.sqrt(t) 
        - C_2 * (1 - np.exp(t / tau) * erfc(np.sqrt(t / tau))))


def deltaT_depth_pstd(t, N, pen_depth, k, rho, C, m_b, I0, T_tr):
    r"""Numerical solution to 1-D Pennes' bioheat transfer equation by
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
    kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)
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


def deltaT_3d_pstd(t, N, area, pen_depth, k, rho, C, m_b, SAR_sur):
    r"""Numerical solution to 1-D Pennes' bioheat transfer equation by
    using Fast Fourier Transform on spatial coordinate.
    
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
    m_b : float
        blood perfusion
    SAR_sur : numpy.ndarray
        2-D array of shape (`N[0]`, `N[1]`), each value corresponds to
        (x, y) surface SAR value
        
    Returns
    -------
    numpy.ndarray
        temperature distribution in time for each collocation point
    """
    Nx, Ny, Nz = N
    X, Y = area
    Z = pen_depth
    dx, dy, dz = X / Nx, Y / Ny, Z / Nz
    x = np.linspace(-X/2, X/2, Nx)
    y = np.linspace(-Y/2, Y/2, Ny)
    z = np.linspace(0, Z, Nz)

    SAR = np.empty(shape=(Nx, Ny, Nz))
    for idx in range(Nz):
        _SAR = SAR_sur * np.exp(-z[idx] / pen_depth)
        SAR[:, :, idx] = _SAR

    kappax = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    kappay = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kappaz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)

    T0 = np.zeros_like(SAR)
    T0 = T0.ravel()

    def rhs(T, t, kappa, k, rho, C, m_b, SAR):
        T = T.reshape(Nx, Ny, Nz)
        T_fft = np.fft.fft(T)
        dd_T_fft = - np.power(kappa, 2) * T_fft
        dd_T = np.fft.ifft(dd_T_fft)

        dT_dt = (
            k * dd_T / (rho * C)
            - rho * m_b * T
            + SAR / C
            )
        return dT_dt.real.ravel()

    T = odeint(rhs, T0, t, args=(kappax, k, rho, C, m_b, SAR))
    return T.reshape(-1, Nx, Ny, Nz)