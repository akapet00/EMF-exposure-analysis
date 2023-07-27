import jax.numpy as jnp
from jax import jacobian, vmap
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
    if not isinstance(a, (int, float, )):
        raise ValueError('Left boundary must be a number.')
    if not isinstance(b, (int, float, )):
        raise ValueError('Right boundary must be a number.')
    points, w = roots_legendre(degree)
    scaler = (b - a) / 2
    x = scaler * (points + 1.) + a
    val = (scaler * w) @ func(x, *args)
    return val


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
    if not isinstance(bbox, (list, tuple, np.ndarray, )):
        raise ValueError('Integration domain must be iterable.')
    points, w = roots_legendre(degree)
    x_a, x_b, y_a, y_b = bbox
    x_scaler = (x_b - x_a) / 2
    y_scaler = (y_b - y_a) / 2
    x_scaled = x_scaler * (points + 1.) + x_a
    y_scaled = y_scaler * (points + 1.) + y_a
    X, Y = np.meshgrid(x_scaled, y_scaled)
    val = (x_scaler * w) @ func(X, Y, *args) @ (y_scaler * w)
    return val


def surfquad(func, surf, bbox, args=(), degree=9):
    """Return the approximation of the flux of a vector field through a
    2-D surface in bounded 3-D space.

    Parameters
    ----------
    func : callable
        Scalar field which takes single `jax.numpy.ndarray` consisting
        of 3 elements corresponding to x, y, and z coordinate,
        respectively, e.g., `func = lambda X: return jnp.sum(X)`.
    surf : callable
        2-D parameterized surface. It should take a single 2-element
        `jax.numpy.ndarray` as its argument where the first element
        corresonds to the first parametric coordinate, while the second
        element corresponds to the second parametric coordinate. It
        should return 3-element `jax.numpy.ndarray` where its elements
        correspond to x, y, and z coordinate in 3-D space,
        respectively, e.g.,
        `surf = lambda T: return jnp.asarray([T[0], T[1], 0])`.
    bbox : list or tuple
        Bounds of the integration domain in 2-D parameterized space.
    args : tuple, optional
        Additional arguments for `func`.
    degree : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        Surface integral of a given scalar field.
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    if not callable(surf):
        raise ValueError('`surf` must be callable')
    if not isinstance(bbox, (list, tuple, np.ndarray, )):
        raise ValueError('Integration domain must be iterable.')
    points, w = roots_legendre(degree)
    T0_scaler = (bbox[1] - bbox[0]) / 2
    T1_scaler = (bbox[3] - bbox[2]) / 2
    T0 = T0_scaler * (points + 1.) + bbox[0]
    T1 = T1_scaler * (points + 1.) + bbox[2]
    T = jnp.meshgrid(T0, T1)
    T = jnp.column_stack((T[0].ravel(), T[1].ravel()))
    S = vmap(surf, in_axes=0)(T)
    S_jac = vmap(jacobian(surf), in_axes=0)(T)
    n = jnp.cross(S_jac[:, :, 0], S_jac[:, :, 1])
    n_len = jnp.sqrt(n[:, 0] ** 2 + n[:, 1] ** 2 + n[:, 2] ** 2)
    in_axes = [0, ]
    in_axes.extend([None] * len(args))
    F = vmap(func, in_axes=in_axes)(S, *args)
    integrand = (F * n_len).reshape(degree, degree)
    val = (T0_scaler * w) @ integrand @ (T1_scaler * w)
    return val.item()


def flux(func, surf, bbox, args=(), degree=9):
    """Return the approximation of the flux of a vector field through a
    2-D surface in bounded 3-D space.

    Parameters
    ----------
    func : callable
        Vector field which takes single `jax.numpy.ndarray` consisting
        of 3 elements corresponding to x, y, and z coordinate,
        respectively, e.g., `func = lambda X: return jnp.asarray(X)`.
    surf : callable
        2-D parameterized surface. It should take a single 2-element
        `jax.numpy.ndarray` as its argument where the first element
        corresonds to the first parametric coordinate, while the second
        element corresponds to the second parametric coordinate. It
        should return 3-element `jax.numpy.ndarray` where its elements
        correspond to x, y, and z coordinate in 3-D space,
        respectively, e.g.,
        `surf = lambda T: return jnp.asarray([T[0], T[1], 0])`.
    bbox : list or tuple
        Bounds of the integration domain in 2-D parameterized space.
    args : tuple, optional
        Additional arguments for `func`.
    degree : int, optional
        Degree of the Gauss-Legendre quadrature.

    Returns
    -------
    float
        Surface integral of a given vector field.
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    if not callable(surf):
        raise ValueError('`surf` must be callable')
    if not isinstance(bbox, (list, tuple, np.ndarray, )):
        raise ValueError('Integration domain must be iterable.')
    points, w = roots_legendre(degree)
    T0_scaler = (bbox[1] - bbox[0]) / 2
    T1_scaler = (bbox[3] - bbox[2]) / 2
    T0 = T0_scaler * (points + 1.) + bbox[0]
    T1 = T1_scaler * (points + 1.) + bbox[2]
    T = jnp.meshgrid(T0, T1)
    T = jnp.column_stack((T[0].ravel(), T[1].ravel()))
    S = vmap(surf, in_axes=0)(T)
    S_jac = vmap(jacobian(surf), in_axes=0)(T)
    n = jnp.cross(S_jac[:, :, 0], S_jac[:, :, 1])
    in_axes = [0, ]
    in_axes.extend([None] * len(args))
    F = vmap(func, in_axes=in_axes)(S, *args)
    integrand = jnp.sum(F * n, axis=1).reshape(degree, degree)
    val = (T0_scaler * w) @ integrand @ (T1_scaler * w)
    return val.item()


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
    if not isinstance(values, (jnp.ndarray, np.ndarray, )):
        raise ValueError('`values` must be array-like.')
    try:
        a = points[0].item()
        b = points[-1].item()
    except Exception:
        print('`points` must be array-like')
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
        interpolation is used: `scipy.interpolate.RBFInterpolator`.
    kwargs : dict, optional
        Additional keyword arguments for the interpolation function.

    Returns
    -------
    float
        Approximation of the integral for givend 2-D data.
    """
    if not isinstance(values, (jnp.ndarray, np.ndarray, )):
        raise Exception('`values` must be array-like.')
    try:
        bbox = [points[:, 0].min().item(), points[:, 0].max().item(),
                points[:, 1].min().item(), points[:, 1].max().item()]
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
    if not isinstance(values, (jnp.ndarray, np.ndarray, )):
        raise Exception('`values` must be array-like.')
    try:
        bbox = [x.min(), x.max(), y.min(), y.max()]
    except TypeError:
        print('Both `x` and `y` must be arrays')
    func = interpolate.RectBivariateSpline(x, y, values, bbox=bbox, **kwargs)
    val = func.integral(*bbox)
    return val


def elementwise_circquad(points, values, radius, center, degree=9,
                         interp_func=None, **kwargs):
    """Return the approximate value of the integral for given sampled
    2-D data over a disk by using the appropriate quadrature scheme for
    a given degree of integration.

    Note: DEPRECATED

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
        interpolation is used: `scipy.interpolate.Rbf`.
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
    val = scheme.integrate(f=lambda x: func(x[0], x[1]),
                           center=center, radius=radius, dot=np.matmul)
    return val
