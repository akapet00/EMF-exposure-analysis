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
    xt : float
        x coordinate of the observed point in free space.
    yt : float
        y coordinate of the observed point in free space.
    zt : float
        z coordinate of the observed point in free space.
    xs : float or numpy.ndarray
        x coordinates) of the source.
    xs : float or numpy.ndarray
        y coordinates of the source.
    xs : float or numpy.ndarray
        z coordinates of the source.
    f : float
        Frequency in GHz.

    Returns
    -------
    float or numpy.ndarray or jax.numpy.ndarray
        Green function value for the observed point(s).
    """
    omega = 2 * pi * f
    k = jnp.sqrt(omega ** 2 * eps_0 * mu_0)
    R = jnp.sqrt((xt - xs) ** 2 + (yt - ys) ** 2 + (zt - zs) ** 2)
    return jnp.exp(-1j * k * R) / R


# just-in-time compilation of the Green function
green = jit(green)


# autodiff
green_grad = jit(vmap(
        grad(green, argnums=(0, 1, 2), holomorphic=True),
        in_axes=(None, None, None, 0, 0, 0, None)))


def efield(xt, yt, zt, xs, ys, zs, Is, f):
    """Return electric field components for a single point in free
    space.

    Parameters
    ----------
    xt : float
        x-coordinate of the target point.
    yt : float
        y-coordinate of the target point.
    zt : float
        z-coordinate of the target point.
    xs : array-like
        x-coordinates of the source.
    xs : array-like
        y-coordinates of the source.
    xs : array-like
        z-coordinates of the source.
    Is : array-like
        Complex current distribution along the antenna.
    f : float
        Frequency in GHz.

    Returns
    -------
    tuple
        Electric field x-, y-, and z-component.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    dx = xs[1] - xs[0]
    Is_x = jnp.asarray(holoborodko(Is, dx))
    prefix = 1 / (1j * 4 * pi * omega * eps_0)
    g = green(xt, yt, zt, xs, ys, zs, f)
    g_x, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs, f)
    Ex = prefix * (- equad(xs, Is_x * g_x, 3)
                   - gamma ** 2 * equad(xs, Is * g, 3))
    Ey = prefix * (equad(xs, Is_x * g_y, 3))
    Ez = prefix * (equad(xs, Is_x * g_z, 3))
    return Ex.item(), Ey.item(), Ez.item()


def hfield(xt, yt, zt, xs, ys, zs, Is, f):
    """Return magnetic field components for a single point in free
    space.

    Parameters
    ----------
    xt : float
        x-coordinate of the target point.
    yt : float
        y-coordinate of the target point.
    zt : float
        z-coordinate of the target point.
    xs : array-like
        x-coordinates of the source.
    xs : array-like
        y-coordinates of the source.
    xs : array-like
        z-coordinates of the source.
    Is : array-like
        Complex current distribution along the antenna.
    f : float
        Frequency in GHz.

    Returns
    -------
    tuple
        Magnetic field x-, y-, and z-component.
    """
    prefix = 1 / (4 * pi)
    _, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs, f)
    Hy = prefix * equad(xs, Is * g_z, 3)
    Hz = - prefix * equad(xs, Is * g_y, 3)
    Hx = jnp.zeros_like(Hz)
    return Hx.item(), Hy.item(), Hz.item()


def poynting(xt, yt, zt, xs, ys, zs, f, Is, Is_x=None):
    """Return the Poynting vector components for a single point in free
    space.

    Parameters
    ----------
    xt : float
        x-coordinate of the target point.
    yt : float
        y-coordinate of the target point.
    zt : float
        z-coordinate of the target point.
    xs : array-like
        x-coordinates of the source.
    xs : array-like
        y-coordinates of the source.
    xs : array-like
        z-coordinates of the source.
    f : float
        Frequency in GHz.
    Is : array-like
        Complex current distribution along the antenna.
    Is_x : array-like, optional
        First derivative of a current distribution.

    Returns
    -------
    tuple
        The Poynting vector x-, y-, and z-component.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    if Is_x is None:
        dx = xs[1] - xs[0]
        Is_x = jnp.asarray(holoborodko(Is, dx))

    g = green(xt, yt, zt, xs, ys, zs, f)
    g_x, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs, f)

    e_prefix = 1 / (1j * 4 * pi * omega * eps_0)
    Ex = e_prefix * (- equad(xs, Is_x * g_x, 3)
                     - gamma ** 2 * equad(xs, Is * g, 3))
    Ey = e_prefix * (equad(xs, Is_x * g_y, 3))
    Ez = e_prefix * (equad(xs, Is_x * g_z, 3))

    h_prefix = 1 / (4 * pi)
    Hy = h_prefix * equad(xs, Is * g_z, 3)
    Hz = - h_prefix * equad(xs, Is * g_y, 3)

    Sx = Ey * Hz.conjugate() - Ez * Hy.conjugate()
    Sy = Ex * Hz.conjugate()
    Sz = Ex * Hy.conjugate()
    return Sx, Sy, Sz


def poynting_parallel(xt, yt, zt, xs, ys, zs, f, Is, Is_x=None):
    """Return the Poynting vector components for a single point in free
    space.

    Note: work in progress. Not ready for use in real applications
    because of memory-related issues.

    Parameters
    ----------
    xt : float
        x-coordinate of the target point.
    yt : float
        y-coordinate of the target point.
    zt : float
        z-coordinate of the target point.
    xs : array-like
        x-coordinates of the source.
    xs : array-like
        y-coordinates of the source.
    xs : array-like
        z-coordinates of the source.
    f : float
        Frequency in GHz.
    Is : array-like
        Complex current distribution along the antenna.
    Is_x : array-like, optional
        First derivative of a current distribution.

    Returns
    -------
    tuple
        The Poynting vector x-, y-, and z-components.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    if Is_x is None:
        dx = xs[1] - xs[0]
        Is_x = jnp.asarray(holoborodko(Is, dx))

    g_vmap_z = vmap(green, in_axes=(None, None, 0, None, None, None, None))
    g_vmap_yz = vmap(g_vmap_z, in_axes=(None, 0, None, None, None, None, None))
    g_vmap_xyz = vmap(g_vmap_yz, in_axes=(0, None, None, None, None, None,
                                          None))
    g = jnp.stack(
        g_vmap_xyz(xt, yt, zt, xs, ys, zs, f)
        ).reshape(xt.size * yt.size * zt.size, xs.size)

    g_grad_vmap_z = vmap(green_grad, in_axes=(None, None, 0, None, None, None,
                                              None))
    g_grad_vmap_yz = vmap(g_grad_vmap_z, in_axes=(None, 0, None, None, None,
                                                  None, None))
    g_grad_vmap_xyz = vmap(g_grad_vmap_yz, in_axes=(0, None, None, None, None,
                                                    None, None))
    g_grad = jnp.stack(
        g_grad_vmap_xyz(xt, yt, zt, xs, ys, zs, f), axis=3
        ).reshape(xt.size * yt.size * zt.size, xs.size)

    e_prefix = 1 / (1j * 4 * pi * omega * eps_0)

    Sx, Sy, Sz = []
    loops = g.shape[0]
    for i in loops:
        _g = g[i, :]
        _g_x = g_grad[i, 0, :]
        _g_y = g_grad[i, 1, :]
        _g_z = g_grad[i, 2, :]
        Ex = e_prefix * (- equad(xs, Is_x * _g_x, 3)
                         - gamma ** 2 * equad(xs, Is * _g, 3))
        Ey = e_prefix * (equad(xs, Is_x * _g_y, 3))
        Ez = e_prefix * (equad(xs, Is_x * _g_z, 3))

        h_prefix = 1 / (4 * pi)
        Hy = h_prefix * equad(xs, Is * _g_z, 3)
        Hz = - h_prefix * equad(xs, Is * _g_y, 3)

        Sx.append(Ey * Hz.conjugate() - Ez * Hy.conjugate())
        Sy.append(Ex * Hz.conjugate())
        Sz.append(Ex * Hy.conjugate())
    return Sx, Sy, Sz
