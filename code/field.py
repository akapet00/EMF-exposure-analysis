import jax.numpy as jnp
from jax import (grad, jit, vmap)
from jax.ops import index, index_update
import numpy as np

from .constants import (eps_0, mu_0, pi, c)
from .utils.integrate import elementwise_quad as equad
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
    Ex = prefix * (- equad(Is_x * g_x, xs, 3)
                   - gamma ** 2 * equad(Is * g, xs, 3))
    Ey = prefix * (equad(Is_x * g_y, xs, 3))
    Ez = prefix * (equad(Is_x * g_z, xs, 3))
    return (Ex, Ey, Ez)


def hfield(xt, yt, zt, xs, ys, zs, Is, frequency):
    r"""Return the magnetic field approximation value in a single point
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
        magnetic field values for the observed point
    """
    prefix = 1 / (4 * pi)
    g_x, g_y, g_z = green_grad(xt + 0j, yt + 0j, zt + 0j, xs, ys, zs,
                               frequency)
    Hy = prefix * equad(Is * g_z, xs, 3)
    Hz = - prefix * equad(Is * g_y, xs, 3)
    Hx = np.zeros_like(Hz)
    return (Hx, Hy, Hz)


def incident_power_density_ff(efield):
    r"""Return the incident power density for the case of the far-field
    or transverse electromagnetic plane wave based on absolute values
    of electric field.

    Parameters
    ----------
    efield : float or numpy.ndarray
        electric field absolute value or multiple values

    Returns
    -------
    numpy.ndarray
        incidnet power density free space approximation of shape (N, )
        where N is the shape of `efield`
    """
    Z_0 = mu_0 * c
    if isinstance(efield, (float, jnp.ndarray, np.ndarray)):
        return efield ** 2 / Z_0
    else:
        raise ValueError('Data type for `efield` should be either float or '
                         'numpy.ndarray.')


def power_density(xt, yt, zt, xs, ys, zs, Is, frequency):
    r"""Return the incident power density for the case of the
    near-field wave incidence based on the distribution of electric and
    magnetic field in space.

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
    tuple
        containing power density and incident power density, both items
        are 3-D arrays of shape (`xt.size`, `yt.size`, `zt.size`)
    """
    omega = 2 * pi * frequency
    gamma = 1j * jnp.sqrt(omega ** 2 * mu_0 * eps_0)
    dx = xs[1] - xs[0]
    Is_x = holoborodko(Is, dx)
    prefix_e = 1 / (1j * 4 * pi * omega * eps_0)
    prefix_h = 1 / (4 * pi)
    I0 = jnp.empty((xt.size, yt.size, zt.size))
    S0 = jnp.empty_like(I0)
    for x_idx, _xt in enumerate(xt):
        for y_idx, _yt in enumerate(yt):
            for z_idx, _zt in enumerate(zt):
                g = green(_xt, _yt, _zt, xs, ys, zs, frequency)
                g_x, g_y, g_z = green_grad(_xt + 0j, _yt + 0j, _zt + 0j, xs,
                                           ys, zs, frequency)
                Ex = prefix_e * (- equad(Is_x * g_x, xs, 3)
                                 - gamma ** 2 * equad(Is * g, xs, 3))
                Ey = prefix_e * (equad(Is_x * g_y, xs, 3))
                Ez = prefix_e * (equad(Is_x * g_z, xs, 3))

                Hy = prefix_h * equad(Is * g_z, xs, 3)
                Hz = - prefix_h * equad(Is * g_y, xs, 3)
                I0m = jnp.sqrt(np.power(Ey * Hz.conj() - Ez * Hy.conj(), 2)
                               + np.power(Ex * Hz.conj(), 2)
                               + np.power(Ex * Hy.conj(), 2))
                I0 = index_update(I0, index[x_idx, y_idx, z_idx],
                                  jnp.abs(I0m))
                S0 = index_update(S0, index[x_idx, y_idx, z_idx],
                                  1 / 2 * jnp.real(I0m))
    return (S0, I0)
