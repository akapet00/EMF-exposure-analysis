import jax.numpy as jnp
import numpy as np
from scipy import interpolate
from scipy.special import roots_legendre


def quad(func, a, b, args=(), degree=3):
    """Return the integral of a given function solved by means of the
    Gauss-Legendre quadrature.

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
    degree : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        integral of a given function
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    points, w = roots_legendre(degree)
    scaler = (b - a) / 2
    x = scaler * (points + 1.) + a
    I_approx = (scaler * w) @ func(x, *args)
    return I_approx


def dblquad(func, bbox, args=(), degree=9):
    """Return the integral of a given 2-D function, `f(x, y)`, solution
    of which is carried by using the Gauss-Legendre quadrature in 2-D.

    Parameters
    ----------
    func : callable
        Integrand function.
    bbox : list or tuple
        Integration domain [min(x), max(x), min(y), max(y)].
    args : tuple, optional
        Additional arguments for `func`.
    degree : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        Integral of a given function.
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    points, w = roots_legendre(degree)
    x_a, x_b, y_a, y_b = bbox
    x_scaler = (x_b - x_a) / 2
    y_scaler = (y_b - y_a) / 2
    x_scaled = x_scaler * (points + 1.) + x_a
    y_scaled = y_scaler * (points + 1.) + y_a
    X, Y = np.meshgrid(x_scaled, y_scaled)
    I_approx = (x_scaler * w) @ func(X, Y, *args) @ (y_scaler * w)
    return I_approx


def elementwise_quad(points, values, degree=3, interp_func=None, **kwargs):
    """Return the approximate value of the integral for given sampled
    data using the Gauss-Legendre quadrature.

    Parameters
    ----------
    points : numpy.ndarray
        Integration domain.
    values : numpy.ndarray
        Sampled integrand.
    degree : int, optional
        Degree of the Gauss-Legendre quadrature.
    interp_func : callable, optional
        Interpolation function. The default is
        `scipy.interpolate.interp1d`.
    kwargs : dict, optional
        Additional keyword arguments for the interpolation function.

    Returns
    -------
    float
        Approximation of the integral for given data.
    """
    if not isinstance(values, (jnp.ndarray, np.ndarray)):
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
    return quad(func, a, b, degree=degree)


def elementwise_dblquad(points, values, degree=9, interp_func=None, **kwargs):
    """Return the approximate value of the integral for sampled 2-D
    data by using the Gauss-Legendre quadrature in 2-D.

    Parameters
    ----------
    points : numpy.ndarray
        Data points of shape (n, 2), where n is the number of points.
    values : numpy.ndarray
        Sampled integrand function values of shape (n, ). If the data
        is sampled over a grid it could also be of shape (m, m), where
        m corresponds to the number of data points coordinates.
    degree : int, optional
        Degree of the Gauss-Legendre quadrature.
    interp_func : callable, optional
        Interpolation function. If not set radial basis function
        interpolation is used.
    kwargs : dict, optional
        Additional keyword arguments for the interpolation function.

    Returns
    -------
    float
        Approximation of the integral for givend 2-D data.
    """
    if not isinstance(values, (np.ndarray, np.ndarray)):
        raise Exception('`values` must be array-like.')
    try:
        bbox = [points[:, 0].min(), points[:, 0].max(),
                points[:, 1].min(), points[:, 1].max()]
    except TypeError:
        print('`points` must be a 2-column array.')
    if interp_func is None:
        func = interpolate.Rbf(points[:, 0], points[:, 1], values, **kwargs)
    else:
        func = interp_func(points, values, **kwargs)
    return dblquad(func, bbox, degree=degree)


def elementwise_rectquad(x, y, values, **kwargs):
    """Return the approximate value of the integral for given sampled
    2-D data over a rectangular grid.

    Parameters
    ----------
    x : numpy.ndarray
        x-axis strictly ascending coordinates.
    y : numpy.ndarray
        y-axis strictly ascending coordinates.
    values: numpy.ndarray
        Sampled integrand function of shape (`x.size`, `y.size`).
    kwargs : dict, optional
        Additional keyword arguments for
        `scipy.interpolate.RectBivariateSpline` function.

    Returns
    -------
    float
        Approximation of the integral of a given function.
    """
    if not isinstance(values, (jnp.ndarray, np.ndarray)):
        raise Exception('`values` must be array-like.')
    try:
        bbox = [x.min(), x.max(), y.min(), y.max()]
    except TypeError:
        print('Both `x` and `y` must be arrays')
    func = interpolate.RectBivariateSpline(x, y, values, bbox=bbox, **kwargs)
    I_approx = func.integral(*bbox)
    return I_approx


def elementwise_circquad(points, values, radius, center, degree=9,
                         interp_func=None, **kwargs):
    """Return the approximate value of the integral for given sampled
    2-D data over a disk by using the appropriate quadrature scheme for
    a given degree of integration.

    Parameters
    ----------
    points : numpy.ndarray
        Data point coordinates of shape (n, 2) on a disk.
    values : numpy.ndarray
        Sampled integrand function values of shape (n, ).
    radius : float
        Radius of the integration domain.
    center : list or numpy.ndarray
        x- and y-coordinate of the center of the integration domain.
    degree : int, optional
        Degree of the quadrature. Should be less or equal to 21.
    interp_func : callable, optional
        Interpolation function. If not set, radial basis function
        interpolation is used.
    kwargs : dict, optional
        Additional keyword arguments for the interpolation function.

    Returns
    -------
    float
        Approximation of the integral.
    """
    try:
        import quadpy
    except ImportError:
        raise ImportError('`quadpy` is not installed.')
    degree = int(degree)
    if degree > 21:
        raise ValueError('Highest integration order is 21.')
    if not isinstance(values, (jnp.ndarray, np.ndarray)):
        raise Exception('`values` must be array-like.')
    points = np.atleast_2d(points)
    if (interp_func is None) or (interp_func is interpolate.Rbf):
        func = interpolate.Rbf(points[:, 0], points[:, 1], values, **kwargs)
    else:
        func = interp_func(points, values, **kwargs)
    scheme = quadpy.s2.get_good_scheme(degree)
    I_approx = scheme.integrate(f=lambda x: func(x[0], x[1]),
                                center=center, radius=radius, dot=np.matmul)
    return I_approx
