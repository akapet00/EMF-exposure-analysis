import os

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from dosipy.constants import eps_0
from dosipy.field import poynting
from dosipy.utils.dataloader import (load_tissue_diel_properties,
                                     load_antenna_el_properties)
from dosipy.utils.derive import holoborodko

from utils import reflection_coefficient


def main():
    # working frequency, Hz
    f = np.array([10, 30, 90]) * 1e9

    # distance from the antenna
    d_10_30 = np.array([5, 10, 15]) / 1000  # meters
    d_90 = np.array([2, 5, 10]) / 1000  # meters

    # dry skin density, kg/m3
    rho = 1109

    # conductivity, S/m
    sigma = np.array([8.48, 27.31, 41.94])

    # dielectric constant
    eps_r = np.array([32.41, 16.63, 6.83])

    # reflection coefficient
    eps_i = sigma / (2 * np.pi * f * eps_0)
    eps = eps_r - 1j * eps_i
    gamma = reflection_coefficient(eps)

    # power transmission coefficient
    T_tr = 1 - gamma ** 2

    # exposed surface extent
    exposure_extent = (0.02, 0.02)  # meters x meters

    # exposed volume coordinates
    xt = jnp.linspace(-exposure_extent[0]/2, exposure_extent[0]/2)
    yt = jnp.linspace(-exposure_extent[1]/2, exposure_extent[1]/2)
    z_max = 0.02  # in meters
    zt = jnp.linspace(0, z_max)
    
    for _f in tqdm(f):
        # antenna electric properties, free space (Poljak 2005)
        dipole_props = load_antenna_el_properties(_f)

        # antenna position - coordinates
        xs = dipole_props.x.to_numpy()
        xs = xs - xs.max() / 2
        xs = jnp.asarray(xs)
        ys = jnp.zeros_like(xs)

        # current through the antenna
        Is = dipole_props.ireal.to_numpy() + dipole_props.iimag.to_numpy() * 1j

        # current gradients
        Is_x = holoborodko(Is, xs[1]-xs[0])

        if _f/1e9 in [10, 30]:
            d = d_10_30
        else:
            d = d_90
        for _d in tqdm(d):
            zs = jnp.full_like(xs, _d)
            _zt = zt[0]

            # incident PD components on the exposed surface
            PDinc = np.empty((xt.size, yt.size, 3), dtype=np.complex128)
            for xi, _xt in enumerate(xt):
                for yi, _yt in enumerate(yt):
                    PDinc[xi, yi, :] = poynting(_xt, _yt, _zt,
                                                xs, ys, zs,
                                                _f, Is, Is_x)
            file = os.path.basename(__file__).removesuffix('.py')
            fname = f'{fname}_d{int(_d*1000)}mm_f{int(_f/1e9)}GHz'
            np.save(os.path.join('data', fname), PDinc)
        

if __name__ == '__main__':
    main()
