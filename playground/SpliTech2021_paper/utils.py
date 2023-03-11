import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


# differentiation
def central_difference(func, axis='x', args=(), order=1, delta=1.e-4):
    r"""Return n-th order central numerical difference of a given
    time-independent function.

    If order is not given, it is assumed to be 1.

    Parameters
    ----------
    func : callable
        function to derive w.r.t. a single variable
    axis : string, optional
        differentiation domain
    args : tuple, optional)
        additional arguments of a function
    order : int, optional
        numerical derivation order
    delta : float, optional
        numerical derivation precision

    Returns
    -------
    numpy.ndarray
        central difference of func
    """
    if axis not in ['x', 'y', 'z']:
        raise ValueError('`x`, `y` and `z` axis are supported.')
    if order not in [1, 2]:
        raise ValueError(f'Differentiation order {order} is not supported.')
    precision_low = 1.0e-1
    precision_high = 1.0e-16
    if delta > precision_low:
        raise ValueError(f'`delta` has to be smaller than {precision_low}.')
    elif delta < precision_high:
        raise ValueError(f'`delta` has to be larger than {precision_high}.')
    if axis == 'x':
        def f(x):
            if order == 1:
                return (func(x + delta, *args)
                        - func(x - delta, *args)) / (2 * delta)
            if order == 2:
                return (func(x + delta, *args)
                        - 2 * func(x, *args)
                        + func(x - delta, *args)) / delta ** 2
    elif axis == 'y':
        def f(y):
            if order == 1:
                return (func(args[0], y + delta, *args[1:])
                        - func(args[0], y - delta, *args[1:])) / (2 * delta)
            if order == 2:
                return (func(args[0], y + delta, *args[1:])
                        - 2 * func(args[0], y, *args[1:])
                        + func(args[0], y - delta, *args[1:])) / delta ** 2
    else:
        def f(z):
            if order == 1:
                return (func(*args[:2], z + delta, *args[2:])
                        - func(*args[:2], z - delta, *args[2:])) / (2 * delta)
            if order == 2:
                return (func(*args[:2], z + delta, *args[2:])
                        - 2 * func(*args[:2], z, *args[2:])
                        + func(*args[:2], z - delta, *args[2:])) / delta ** 2
    return f


# visualization
def fig_config(latex=False, nrows=1, ncols=1, scaler=1.0):
    r"""Configure matplotlib parameters for better visualization style.

    Parameters
    ----------
    latex : bool, optional
        If true, LaTeX backend will be used
    nrows : int, optional
        number of figures row-wise
    ncols : int, optional
        number of figures column-wise
    scaler : float, optional
        scaler for each figure

    Returns
    -------
    None
    """
    plt.rcParams.update({
        'text.usetex': latex,
        'font.family': 'serif',
        'font.size': 14,
        'figure.figsize': (4.774 * scaler * ncols, 2.950 * scaler * nrows),
        'lines.linewidth': 3,
        'lines.dashed_pattern': (3, 5),
        'lines.markersize': 10,
        'lines.markeredgecolor': 'k',
        'lines.markeredgewidth': 0.5,
        'image.origin': 'lower',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'grid.linewidth': 0.5,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })


def fig_config_reset():
    r"""Recover matplotlib default parameters.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    plt.rcParams.update(plt.rcParamsDefault)


# error evaulation
def mse(true, pred):
    r"""Return mean square difference between two values.

    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)

    Returns
    -------
    float
        Root mean square error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.mean((true - pred) ** 2)


def rmse(true, pred):
    r"""Return root mean square difference between two values.

    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)

    Returns
    -------
    float
        Root mean square error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.sqrt(mse(true, pred))


def msle(true, pred):
    r"""Return mean square log difference between two values.

    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)

    Returns
    -------
    float
        Mean square log error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return mse(np.log1p(true), np.log1p(pred))


def mae(true, pred):
    r"""Return mean absolute difference between two values.

    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)

    Returns
    -------
    float
        Mean absolute error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.mean(np.abs(true - pred))
