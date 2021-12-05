import collections

import jax.numpy as jnp
import numpy as np
from scipy import interpolate
from scipy.special import roots_legendre


def quad(func, a, b, args=(), leggauss_deg=3):
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
    leggauss_deg : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        integral of a given function
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    points, w = roots_legendre(leggauss_deg)
    scaler = (b - a) / 2
    x = scaler * (points + 1.) + a
    I_num = (scaler * w) @ func(x, *args)
    return I_num


def dblquad(func, bbox, args=(), leggauss_deg=9):
    """Return the the integral of a given 2-D function, `f(x, y)`,
    using the Gauss-Legendre quadrature scheme.

    Parameters
    ----------
    func : callable
        Integrand function.
    bbox : list or tuple
        Integration domain [min(x), max(x), min(y), max(y)].
    args : tuple, optional
        Additional arguments for `func`.
    leggauss_deg : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        Integral of a given function.
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    points, w = roots_legendre(leggauss_deg)
    x_a, x_b, y_a, y_b = bbox
    x_scaler = (x_b - x_a) / 2
    y_scaler = (y_b - y_a) / 2
    x_scaled = x_scaler * (points + 1.) + x_a
    y_scaled = y_scaler * (points + 1.) + y_a
    X, Y = np.meshgrid(x_scaled, y_scaled)
    I_num = (x_scaler * w) @ func(X, Y, *args) @ (y_scaler * w)
    return I_num


def elementwise_quad(points, values, leggauss_deg=3, interp_func=None,
                     **kwargs):
    """Return the approximate value of the integral of a given sampled
    data using the Gauss-Legendre quadrature.

    Parameters
    ----------
    points : numpy.ndarray
        Integration domain.
    values : numpy.ndarray
        Sampled integrand.
    leggauss_deg : int, optional
        Degree of the Gauss-Legendre quadrature.
    interp_func : callable, optional
        Interpolation function.
    kwargs : dict, optional
        Additional keyword arguments for interpolation function

    Returns
    -------
    float
        Approximation of the integral of a given function.
    """
    if not isinstance(values, (collections.Sequence, jnp.ndarray, np.ndarray)):
        raise Exception('`y` must be array-like.')
    try:
        a = points[0]
        b = points[-1]
    except TypeError:
        print('`x` must be array-like')
    if interp_func is None:
        func = interpolate.interp1d(points, values, **kwargs)
    else:
        func = interp_func(points, values, **kwargs)
    return quad(func, a, b, leggauss_deg=leggauss_deg)


def elementwise_dblquad(points, values, leggaus_deg=9, interp_func=None,
                        **kwargs):
    """Return the approximate value of the integral of a given sampled
    2-D data using the Gauss-Legendre quadrature.

    Parameters
    ----------
    points : numpy.ndarray
        Data point coordinates of shape (n, D).
    values : numpy.ndarray
        Sampled integrand function values of shape (n, ). If the data
        is sampled over a grid data, it could also be of shape (m, m),
        where m corresponds to the number of data points coordinates.
    leggauss_deg : int, optional
        Degree of the Gauss-Legendre quadrature.
    interp_func : callable, optional
        Interpolation function. If not set radial basis function
        interpolation is used.
    kwargs : dict, optional
        Additional keyword arguments for interpolation function.

    Returns
    -------
    float
        Approximation of the integral of a given function.
    """
    if not isinstance(values, (collections.Sequence, jnp.ndarray, np.ndarray)):
        raise Exception('`y` must be array-like.')
    try:
        bbox = [points[:, 0].min(), points[:, 0].max(),
                points[:, 1].min(), points[:, 1].max()]
    except TypeError:
        print('Both `x` and `y` must be arrays')
    if interp_func is None:
        func = interpolate.Rbf(points[:, 0], points[:, 1], values, **kwargs)
    else:
        func = interp_func(points[:, 0], points[:, 1], values, **kwargs)
    return dblquad(func, bbox, leggauss_deg=leggaus_deg)


def elementwise_rectquad(x, y, values, leggauss_deg=9, **kwargs):
    """Return the approximate value of the integral of a given sampled
    2-D data over a rectangular grid using the Gauss-Legendre
    quadrature.

    Parameters
    ----------
    x : numpy.ndarray
        x-axis strictly ascending coordinates.
    y : numpy.ndarray
        y-axis strictly ascending coordinates.
    values: numpy.ndarray
        Sampled integrand function of shape (x.size, y.size)
    leggauss_deg : int, optional
        Degree of the Gauss-Legendre quadrature.
    kwargs : dict, optional
        Additional keyword arguments for
        `scipy.interpolate.RectBivariateSpline` function.

    Returns
    -------
    float
        Approximation of the integral of a given function.
    """
    if not isinstance(values, (collections.Sequence, jnp.ndarray, np.ndarray)):
        raise Exception('`values` must be array-like.')
    try:
        bbox = [x.min(), x.max(), y.min(), y.max()]
    except TypeError:
        print('Both `x` and `y` must be arrays')
    func = interpolate.RectBivariateSpline(x, y, values, bbox=bbox, **kwargs)
    I_approx = func.integral(*bbox)
    return I_approx
