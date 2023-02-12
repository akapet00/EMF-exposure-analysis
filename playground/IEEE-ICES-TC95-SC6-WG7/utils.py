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
