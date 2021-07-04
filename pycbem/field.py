"""Electromagnetic field in free space."""

import jax.numpy as jnp
from jax import grad, jit, vmap

from .constants import eps_0, mu_0, pi
from .utils.integrate import elementwise_quad as equad
from .utils.derive import holoborodko


def green(xt, yt, zt, xs, ys, zs, f):
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
    f : float
        frequency in GHz

    Returns
    -------
    float or numpy.ndarray
        Green function value for the observed point(s).
    """
    omega = 2 * pi * f
    k = jnp.sqrt(omega ** 2 * eps_0 * mu_0)
    R = jnp.sqrt((xt - xs) ** 2 + (yt - ys) ** 2 + (zt - zs) ** 2)
    return jnp.exp(-1j * k * R) / R


green = jit(green)


green_grad = jit(vmap(
        grad(green, argnums=(0, 1, 2), holomorphic=True),
        in_axes=(None, None, None, 0, 0, 0, None)))


def efield(xt, yt, zt, xs, ys, zs, Is, f):
    """Return the electric field approximation value in a single point
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
    f : float
        frequency in GHz

    Returns
    -------
    tuple
        Values of electric vector field in x, y and z direction for
        given target point.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    dx = xs[1] - xs[0]
    Is_x = holoborodko(Is, dx)
    prefix = 1 / (1j * 4 * pi * omega * eps_0)
    g = green(xt, yt, zt, xs, ys, zs, f)
    g_x, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs, f)
    Ex = prefix * (- equad(Is_x * g_x, xs, 3)
                   - gamma ** 2 * equad(Is * g, xs, 3))
    Ey = prefix * (equad(Is_x * g_y, xs, 3))
    Ez = prefix * (equad(Is_x * g_z, xs, 3))
    return (Ex, Ey, Ez)


def hfield(xt, yt, zt, xs, ys, zs, Is, f):
    """Return the magnetic field approximation value in a single point
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
    f : float
        frequency in GHz

    Returns
    -------
    tuple
        Values of magnetic vector field in x, y and z direction for
        given target point.
    """
    prefix = 1 / (4 * pi)
    _, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs, f)
    Hy = prefix * equad(Is * g_z, xs, 3)
    Hz = - prefix * equad(Is * g_y, xs, 3)
    Hx = jnp.zeros_like(Hz)
    return (Hx, Hy, Hz)


def poynting(xt, yt, zt, xs, ys, zs, Is, f):
    """Return the magnetic field approximation value in a single point
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
    f : float
        frequency in GHz

    Returns
    -------
    tuple
        Values of Poynting vector in x, y and z direction for given
        target point.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    dx = xs[1] - xs[0]
    Is_x = holoborodko(Is, dx)

    g = green(xt, yt, zt, xs, ys, zs, f)
    g_x, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs, f)

    e_prefix = 1 / (1j * 4 * pi * omega * eps_0)
    Ex = e_prefix * (- equad(Is_x * g_x, xs, 3)
                     - gamma ** 2 * equad(Is * g, xs, 3))
    Ey = e_prefix * (equad(Is_x * g_y, xs, 3))
    Ez = e_prefix * (equad(Is_x * g_z, xs, 3))

    h_prefix = 1 / (4 * pi)
    Hy = h_prefix * equad(Is * g_z, xs, 3)
    Hz = - h_prefix * equad(Is * g_y, xs, 3)

    Sx = Ey * Hz.conjugate() - Ez * Hy.conjugate()
    Sy = Ex * Hz.conjugate()
    Sz = Ex * Hy.conjugate()
    return (Sx, Sy, Sz)
