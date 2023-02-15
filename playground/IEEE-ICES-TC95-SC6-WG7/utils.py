import os

import numpy as np


def fraunhofer_distance(f, D=None, reactive=False):
    """Return Fraunhofer distance of the radiative or reactive
    near-field.
    
    Parameters
    ----------
    f : float or numpy.ndarray
        Operating frequency or multiple frequenceis of the radiator.
    D : float or numpy.ndarray, optional
        The largest dimension of the radiator. Only considered for
        radiative near-field zone.
    reactive : bool, optional
        If True, the distance that corresponds to reactive near-field
        is computed.
    
    Returns
    -------
    float
        Fraunhofer distance.
    """
    from scipy.constants import c
    _lambda = c / f  # incident wavelength in meters
    if reactive:
        return _lambda / (2 * np.pi)
    return 2 * D ** 2 / _lambda


def reflection_coefficient(eps, theta_i=0, polarization='parallel'):
    """Return reflection coefficient for oblique plane wave incidence.
    
    Parameters
    ----------
    eps : float
        Relative complex permittivity.
    theta_i : float
        Angle of incidence in °.
    polarization : str
        Either parallel or perpendicular/normal polarization.
    
    Returns
    -------
    float
        Reflection coefficient.
    """
    polarization = polarization.lower()
    SUPPORTED_POLARIZATIONS = ['parallel', 'normal']
    if polarization not in SUPPORTED_POLARIZATIONS:
        raise ValueError(
            f'Unsupported tissue. Choose from: {SUPPORTED_POLARIZATIONS}.'
            )
    scaler = np.sqrt(eps - np.sin(theta_i) ** 2)
    if polarization == 'parallel':
        return np.abs(
            (-eps * np.cos(theta_i) + scaler)
            / (eps * np.cos(theta_i) + scaler)
        )
    return np.abs(
        (np.cos(theta_i) - scaler)
        / (np.cos(theta_i) + scaler)
    )


def load_dipole_data(f):
    """Return the current distribution over a half-wavelength dipole
    antenna for a given frequency.

    Parameters
    ----------
    f : float
        Frequency of the dipole in Hz.

    Returns
    -------
    pandas.DataFrame
        Current distribution over the wire alongside additional
        configuration details.
    """
    import pandas as pd
    from scipy.io import loadmat
    fname = os.path.join('source', 'half-wavelength-dipole-10mW.mat')
    data = loadmat(fname)['output']
    df = pd.DataFrame(data, columns=['N', 'f', 'L', 'v', 'x', 'Ir', 'Ii'])
    df = df[df.f == f].reset_index(drop=True)
    return df


def estimate_apd(d, f, extent, sigma, eps_r):
    """Return estimated absorbed power density for a given separation
    distance between the dipole and the dry skin target and operating
    frequency of the dipole.
    
    Ref: ICNIRP 2020
    
    Parameters
    ----------
    d : int
        Dipole-to-skin separation distance in mm.
    f : int
        Dipole frequency in GHz.
    extent : tuple
        Extent of the exposed surface. The dipole is placed at the
        center position at `d` mm away from the surface. It should
        containt 4 elements as follows (xmin, xmax, ymin, max) in mm.
    sigma : float
        Tissue conductivity in S/m2.
    eps_r : float
        Relative dielectric tissue permittivity.
    
    Returns
    -------
    tuple
        4 different approximations of the spatially averaged absorbed
        power density in W per m squared by considering normal and norm
        defintion of the incident power density spatially averaged on 1
        and 4 cm squared surface.
    """
    from dosipy.constants import eps_0
    from dosipy.utils.integrate import elementwise_rectquad
    # load data
    PDinc = np.load(os.path.join('data',
                                 f'02_power_density_d{d}mm_f{f}GHz.npy'))
    PDinc_n = PDinc[:, :, 2].real
    PDinc_tot = np.sqrt(PDinc[:, :, 0] ** 2
                        + PDinc[:, :, 1] ** 2
                        + PDinc[:, :, 2] ** 2).real
    lims = (int(PDinc.shape[0]/4),
            int(PDinc.shape[0]*3/4) + 1)
    
    # averaged incident power densities
    A4 = 4 / 100 / 100  # 4 cm2 integration surface
    A1 = A4 / 4  # 1 cm2 integration surface
    x = np.linspace(extent[0], extent[1], PDinc.shape[0]) / 1000
    y = np.linspace(extent[2], extent[3], PDinc.shape[1]) / 1000
    sPDinc_n1 = elementwise_rectquad(x[lims[0]:lims[1]],
                                     y[lims[0]:lims[1]],
                                     PDinc_n[lims[0]:lims[1],
                                             lims[0]:lims[1]]) / (2 * A1)
    sPDinc_tot1 = elementwise_rectquad(x[lims[0]:lims[1]],
                                       y[lims[0]:lims[1]],
                                       PDinc_tot[lims[0]:lims[1],
                                                 lims[0]:lims[1]]) / (2 * A1)
    sPDinc_n4 = elementwise_rectquad(x, y, PDinc_n) / (2 * A4)
    sPDinc_tot4 = elementwise_rectquad(x, y, PDinc_tot) / (2 * A4)
    
    # complex dielectric permitivity given a frequency of the dipole
    eps_i = sigma / (2 * np.pi * f * 1e9 * eps_0)

    # abosolute permittivity
    eps = eps_r - 1j * eps_i

    # skin reflection coefficient
    gamma = reflection_coefficient(eps)

    # power transmission coefficient
    T_tr = 1 - gamma ** 2

    sPDab_n1 = T_tr * sPDinc_n1
    sPDab_tot1 = T_tr * sPDinc_tot1
    sPDab_n4 = T_tr * sPDinc_n4
    sPDab_tot4 = T_tr * sPDinc_tot4
    return sPDab_n1, sPDab_tot1, sPDab_n4, sPDab_tot4
