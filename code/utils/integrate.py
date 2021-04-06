import jax.numpy as np
import numpy as onp
from scipy.interpolate import interp1d


def quad(func, a, b, args=(), n_points=3):
    """Return the the integral of a given function using the
    Gauss-Legendre quadrature scheme.

    Parameters
    ----------
    func : callable
        integrand
    a : float
        left boundary of the integration domain
    b : float
        right boundary of the integration domain
    args : tuple, optional
        additional arguments for `func`
    n_points : int, optional
        degree of the Gauss-Legendre quadrature

    Returns
    -------
    float
        integral of a given function
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    psi, w = onp.polynomial.legendre.leggauss(n_points)
    x = ((b - a) / 2) * psi + (a + b) / 2
    return np.sum((b - a) / 2 * w * func(x, *args))


def elementwise_quad(y, x, n_points=3):
    """Return the approximate value of the integral of a given sampled
    data using the Gauss-Legendre quadrature.

    Parameters
    ----------
    y : numpy.ndarray
        sampled integrand
    x : numpy.ndarray
        integration domain
    n_points : int, optional
        degree of the Gauss-Legendre quadrature

    Returns
    -------
    float
        approximate of the integral of a given function
    """
    if not isinstance(y, (np.ndarray)):
        raise Exception('`y` must be numpy.ndarray.')
    try:
        a = x[0]
        b = x[-1]
    except TypeError:
        print('`x` must be numpy.ndarray')
    psi, w = onp.polynomial.legendre.leggauss(n_points)
    y_interp = interp1d(x, y, kind='cubic')
    x_interp = ((b - a) / 2) * psi + (a + b) / 2
    return np.sum((b - a) / 2 * w * y_interp(x_interp))
