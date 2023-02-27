import os

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from dosipy.constants import eps_0
from dosipy.field import poynting
from dosipy.utils.derive import holoborodko

from utils import reflection_coefficient, load_dipole_data


def main():
    # working frequency, Hz
    f = np.array([10, 30, 90]) * 1e9

    # distance from the antenna
    d_10_30 = np.array([5, 10, 15]) / 1000  # meters
    d_90 = np.array([2, 5, 10]) / 1000  # meters

    # exposed volume coordinates
    xy = jnp.linspace(-0.01, 0.01, 99)
    z = jnp.array([0])
    
    for _f in tqdm(f):
        # antenna electric properties, free space (Poljak 2005)
        dipole_props = load_dipole_data(_f)

        # antenna position - coordinates
        xs = dipole_props.x.to_numpy()
        xs = xs - xs.max() / 2
        xs = jnp.asarray(xs)
        ys = jnp.zeros_like(xs)

        # current through the antenna
        Is = dipole_props.Ir.to_numpy() + dipole_props.Ii.to_numpy() * 1j

        # current gradients
        Is_x = holoborodko(Is, xs[1]-xs[0])

        if _f/1e9 in [10, 30]:
            d = d_10_30.copy()
        else:
            d = d_90.copy()
        for _d in tqdm(d):
            zs = -jnp.full_like(xs, _d)
            zt = z[0]

            # incident PD components on the exposed surface
            PDinc = np.empty((xy.size, xy.size, 3), dtype=np.complex128)
            for xi, xt in enumerate(xy):
                for yi, yt in enumerate(xy):
                    PDinc[xi, yi, :] = poynting(xt, yt, zt,
                                                xs, ys, zs,
                                                _f, Is, Is_x)
            file = os.path.basename(__file__).removesuffix('.py')
            fname = f'{file}_d{int(_d*1000)}mm_f{int(_f/1e9)}GHz'
            np.save(os.path.join('data', fname), PDinc)


if __name__ == '__main__':
    main()
