import jax.numpy as jnp
from jax import (grad, jit, vmap)
import numpy as np

from .constants import (eps_0, mu_0, pi, c)
from .utils.integrate import elementwise_quad
from .utils.derive import holoborodko


def green(xt, yt, zt, xs, ys, zs, frequency):
    """Green's function in free space.

    Parameters
    ----------
    xt : float or numpy.ndarray
        x coordinate of the observed point(s) in free space
    yt : float or numpy.ndarray
        y coordinate of the observed point(s) in free space
    zt : float or numpy.ndarray
        z coordinate of the observed point(s) in free space
    xs : float or numpy.ndarray
        x coordinate of the source
    ys : float or numpy.ndarray
        y coordinate of the source
    zs : float or numpy.ndarray
        z coordinate of the source
    frequency : float
        frequency in GHz

    Returns
    -------
    float or numpy.ndarray
        Green function value for the observed point(s)
    """
    omega = 2 * pi * frequency
    k = jnp.sqrt(omega ** 2 * eps_0 * mu_0)
    R = jnp.sqrt((xt - xs) ** 2 + (yt - ys) ** 2 + (zt - zs) ** 2)
    return jnp.exp(-1j * k * R) / R


green_grad = jit(vmap(
        grad(green, argnums=(0, 1, 2), holomorphic=True),
        in_axes=(None, None, None, 0, 0, 0, None)))


def efield(xt, yt, zt, xs, ys, zs, Is, frequency):
    r"""Return the electric field approximation value in a single point
    in free space.

    Parameters
    ----------
    xt : float or numpy.ndarray
        x coordinate of the observed point(s) in free space
    yt : float or numpy.ndarray
        y coordinate of the observed point(s) in free space
    zt : float or numpy.ndarray
        z coordinate of the observed point(s) in free space
    xs : float or numpy.ndarray
        x coordinate of the source
    xs : float or numpy.ndarray
        y coordinate of the source
    xs : float or numpy.ndarray
        z coordinate of the source
    Is : numpy.ndarray
        coomplex current distribution over the antenna
    frequency : float
        frequency in GHz

    Returns
    -------
    float or numpy.ndarray
        electric field values for the observed point
    """
    omega = 2 * pi * frequency
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    dx = xs[1] - xs[0]
    Is_x = holoborodko(Is, dx)
    prefix = 1 / (1j * 4 * pi * omega * eps_0)
    g = green(xt, yt, zt, xs, ys, zs, frequency)
    g_x, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs,
                               frequency)
    Ex = prefix * (- elementwise_quad(Is_x * g_x, xs, 3)
                   - gamma ** 2 * elementwise_quad(Is * g, xs, 3))
    Ey = prefix * (elementwise_quad(Is_x * g_y, xs, 3))
    Ez = prefix * (elementwise_quad(Is_x * g_z, xs, 3))
    return (Ex, Ey, Ez)


def incident_power_density(efield):
    r"""Return the incident power density free space approximation
    based on absolute vaues of electric field.

    Parameters
    ----------
    efield : float or numpy.ndarray
        electric field absolute value or multiple values

    Returns
    -------
    numpy.ndarray
        incidnet power density free space approximation of shape (N, )
        where N is the shape of `E_fs`
    """
    Z_0 = mu_0 * c
    if isinstance(efield, (float, jnp.ndarray, np.ndarray)):
        return efield ** 2 / Z_0
    else:
        raise ValueError('Data type for `efield` should be either float or '
                         'numpy.ndarray.')
