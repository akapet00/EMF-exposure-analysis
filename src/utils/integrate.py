import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.special import roots_legendre


def quad(func, a, b, args=(), n_points=3):
    """Return the the integral of a given function using the
    Gauss-Legendre quadrature scheme.

    Parameters
    ----------
    func : callable
        Integrand function.
    a : float
        Left boundary of the integration domain.
    b : float
        Right boundary of the integration domain.
    args : tuple, optional
        Additional arguments for `func`.
    n_points : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        integral of a given function
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    psi, w = roots_legendre(n_points)
    xi = ((b - a) / 2) * psi + (a + b) / 2
    return (b - a) / 2 * w @ func(xi, *args)


def dblquad(func, bbox, args=(), n_points=9):
    """Return the the integral of a given 2-D function, `f(y, x)`,
    using the Gauss-Legendre quadrature scheme.

    Parameters
    ----------
    func : callable
        Integrand function.
    a : list or tuple
        Integration domain [min(x), max(x), min(y), max(y)].
    args : tuple, optional
        Additional arguments for `func`.
    n_points : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        Integral of a given function.
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    psi, w = roots_legendre(n_points)
    ay, by, ax, bx = bbox
    xix = (bx - ax) / 2 * psi + (ax + bx) / 2
    xiy = (by - ay) / 2 * psi + (ay + by) / 2
    return (bx - ax) / 2 * (by - ay) / 2 * w @ func(xiy, xix, *args) @ w


def elementwise_quad(y, x, n_points=3):
    """Return the approximate value of the integral of a given sampled
    data using the Gauss-Legendre quadrature.

    Parameters
    ----------
    y : numpy.ndarray
        Sampled integrand.
    x : numpy.ndarray
        Integration domain.
    n_points : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        Approximation of the integral of a given function.
    """
    if not isinstance(y, (np.ndarray, jnp.ndarray)):
        raise Exception('`y` must be numpy.ndarray.')
    try:
        a = x[0]
        b = x[-1]
    except TypeError:
        print('`x` must be numpy.ndarray')
    func = interp1d(x, y, kind='cubic')
    return quad(func, a, b, n_points=n_points)


def elementwise_dblquad(z, x, y, n_points=9):
    """Return the approximate value of the integral of a given sampled
    2-D data using the Gauss-Legendre quadrature.

    Parameters
    ----------
    z: numpy.ndarray
        Sampled integrand function of shape (x.size, y.size)
    y : numpy.ndarray
        y-axis strictly ascending coordinates.
    x : numpy.ndarray
        x-axis strictly ascending coordinates.
    n_points : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        Approximation of the integral of a given function.
    """
    if not isinstance(y, (np.ndarray, jnp.ndarray)):
        raise Exception('`y` must be numpy.ndarray.')
    try:
        bbox = [y[0], y[-1], x[0], x[-1]]
    except TypeError:
        print('Both `x` and `y` must be numpy.ndarray')
    func = RectBivariateSpline(y, x, z, bbox=bbox)
    return dblquad(func, bbox, n_points=n_points)
