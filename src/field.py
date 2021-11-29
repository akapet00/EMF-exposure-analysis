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


green = jit(green)


green_grad = jit(vmap(
        grad(green, argnums=(0, 1, 2), holomorphic=True),
        in_axes=(None, None, None, 0, 0, 0, None)))


def efield(xt, yt, zt, xs, ys, zs, Is, f):
    """Return the electric field approximation value in a single point
    in free space.

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
    Is : jax.numpy.ndarray or numpy.ndarray
        Complex current distribution along the antenna.
    f : float
        Frequency in GHz.

    Returns
    -------
    tuple
        Values of electric vector field in x, y and z direction for
        given target point.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    dx = xs[1] - xs[0]
    Is_x = jnp.asarray(holoborodko(Is, dx))
    prefix = 1 / (1j * 4 * pi * omega * eps_0)
    g = green(xt, yt, zt, xs, ys, zs, f)
    g_x, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs, f)
    Ex = prefix * (- equad(Is_x * g_x, xs, 3)
                   - gamma ** 2 * equad(Is * g, xs, 3))
    Ey = prefix * (equad(Is_x * g_y, xs, 3))
    Ez = prefix * (equad(Is_x * g_z, xs, 3))
    return Ex.item(), Ey.item(), Ez.item()


def hfield(xt, yt, zt, xs, ys, zs, Is, f):
    """Return the magnetic field approximation value in a single point
    in free space.

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
    Is : jax.numpy.ndarray or numpy.ndarray
        Complex current distribution along the antenna.
    f : float
        Frequency in GHz.

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
    return Hx.item(), Hy.item(), Hz.item()


def poynting(xt, yt, zt, xs, ys, zs, f, Is, Is_x=None):
    """Return the magnetic field approximation value in a single point
    in free space.

    Parameters
    ----------
    xt : float
        x coordinate of the observed point(s) in free space.
    yt : float
        y coordinate of the observed point(s) in free space.
    zt : float
        z coordinate of the observed point(s) in free space.
    xs : float or numpy.ndarray
        x coordinates of the source.
    xs : float or numpy.ndarray
        y coordinates of the source.
    xs : float or numpy.ndarray
        z coordinates of the source.
    f : float
        Frequency in GHz.
    Is : numpy.ndarray or jax.numpy.ndarray
        Complex current distribution over the antenna.

    Returns
    -------
    tuple
        Values of Poynting vector in x, y and z direction for given
        target point.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    if Is_x is None:
        dx = xs[1] - xs[0]
        Is_x = jnp.asarray(holoborodko(Is, dx))

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
    return Sx, Sy, Sz


def poynting_parallel(xt, yt, zt, xs, ys, zs, f, Is):
    """Return the magnetic field approximation value in a single point
    in free space.

    Note: Work in progress!

    Parameters
    ----------
    xt : float or jnp.ndarray
        x coordinate(s) of the observed point(s) in free space.
    yt : float or jnp.ndarray
        y coordinate(s) of the observed point(s) in free space.
    zt : float or jnp.ndarray
        z coordinate(s) of the observed point(s) in free space.
    xs : float or numpy.ndarray
        x coordinates of the source.
    xs : float or numpy.ndarray
        y coordinates of the source.
    xs : float or numpy.ndarray
        z coordinates of the source.
    f : float
        Frequency in GHz.
    Is : numpy.ndarray or jax.numpy.ndarray
        Complex current distribution over the antenna.

    Returns
    -------
    tuple
        Values of Poynting vector in x, y and z direction for given
        target point.
    """
    omega = 2 * pi * f
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    dx = xs[1] - xs[0]
    Is_x = jnp.asarray(holoborodko(Is, dx))

    g_vmap_z = vmap(green, in_axes=(None, None, 0, None, None, None, None))
    g_vmap_yz = vmap(g_vmap_z, in_axes=(None, 0, None, None, None, None, None))
    g_vmap_xyz = vmap(g_vmap_yz, in_axes=(0, None, None, None, None, None, None))
    g = jnp.stack(
        g_vmap_xyz(xt, yt, zt, xs, ys, zs, f)
        ).reshape(xt.size * yt.size * zt.size, xs.size)

    g_grad_vmap_z = vmap(green_grad, in_axes=(None, None, 0, None, None, None, None))
    g_grad_vmap_yz = vmap(g_grad_vmap_z, in_axes=(None, 0, None, None, None, None, None))
    g_grad_vmap_xyz = vmap(g_grad_vmap_yz, in_axes=(0, None, None, None, None, None, None))
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
        Ex = e_prefix * (- equad(Is_x * _g_x, xs, 3)
                         - gamma ** 2 * equad(Is * _g, xs, 3))
        Ey = e_prefix * (equad(Is_x * _g_y, xs, 3))
        Ez = e_prefix * (equad(Is_x * _g_z, xs, 3))

        h_prefix = 1 / (4 * pi)
        Hy = h_prefix * equad(Is * _g_z, xs, 3)
        Hz = - h_prefix * equad(Is * _g_y, xs, 3)

        Sx.append(Ey * Hz.conjugate() - Ez * Hy.conjugate())
        Sy.append(Ex * Hz.conjugate())
        Sz.append(Ex * Hy.conjugate())
    return Sx, Sy, Sz
