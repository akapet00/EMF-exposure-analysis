import jax.numpy as jnp


def normalize(ref_data, data):
    """Min-max scaling the features to the [0, 1] range."""
    norm_data = (data - min(ref_data)) / (max(ref_data) - min(ref_data))
    return norm_data


def inv_normalize(norm_data, ref_data):
    """Recovering the initial feature values after min-max scaling."""
    data = min(ref_data) + norm_data * (max(ref_data) - min(ref_data))
    return data


def standardize(ref_data, data):
    """Features are set to have zero-mean and unit-variance."""
    st_data = (data - jnp.mean(ref_data)) / jnp.std(ref_data)
    return st_data


def inv_standardize(st_data, ref_data):
    """Recovering initial feature values after standardization."""
    data = st_data * jnp.std(ref_data) + jnp.mean(ref_data)
    return data


def cart2sph(x, y, z):
    """Return spherical given Cartesain coordinates."""
    r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return r, theta, phi


def sph2cart(r, theta, phi):
    """Return Cartesian given Spherical coordinates."""
    x = r * jnp.cos(phi) * jnp.sin(theta)
    y = r * jnp.sin(phi) * jnp.sin(theta)
    z = r * jnp.cos(theta)
    return x, y, z


def cart2cyl(x, y, z):
    """Return Cylndrical given Cartesain coordinates."""
    r = jnp.sqrt(x ** 2 + y ** 2)
    theta = jnp.arcsin(y / r)
    return r, theta, z


def cyl2cart(r, theta, z):
    """Return Cartesian given Cylndrical coordinates."""
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return x, y, z


def sph_normals(r, theta, phi):
    """Return unit vector field components normal to spherical
    surface."""
    nx = r ** 2 * jnp.cos(phi) * jnp.sin(theta) ** 2 
    ny = r ** 2 * jnp.sin(phi) * jnp.sin(theta) ** 2
    nz = r ** 2 * jnp.cos(theta) * jnp.sin(theta)
    return nx, ny, nz


def cyl_normals(r, theta, z):
    """Return unit vector field components normal to cylndrical
    surface."""
    nx = jnp.cos(theta)
    ny = jnp.sin(theta)
    nz = jnp.zeros_like(z)
    return nx, ny, nz
