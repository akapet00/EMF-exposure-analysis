import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from dosipy.field import poynting
from dosipy.utils.integrate import elementwise_dblquad
from utils import *


def compute_sPDn(f, h, degree):
    # averaging surface area
    if f < 30:
        target_area = (0.02, 0.02)
    else:
        target_area = (0.01, 0.01)
    A = target_area[0] * target_area[1]

    # source
    xs = np.load(os.path.join('data', f'x_at{f}GHz.npy'))
    xs = xs - xs.max() / 2
    xs = jnp.asarray(xs.flatten())
    ys = jnp.zeros_like(xs) + h
    zs = jnp.zeros_like(xs)
    Is = np.load(os.path.join('data', f'current_at{f}GHz.npy'))
    Is = jnp.asarray(Is.flatten())
    dIsdx = jnp.load(os.path.join('data', f'grad_current_at{f}GHz.npy'))
    dIsdx = jnp.asarray(dIsdx.flatten())

    # planar surface
    N = 20
    _x = jnp.linspace(-target_area[0]/2, target_area[0]/2, N)
    _z = jnp.linspace(-target_area[1]/2, target_area[1]/2, N)
    Xt, Zt = jnp.meshgrid(_x, _z)
    xt_pln = Xt.ravel()
    yt_pln = jnp.zeros_like(xt_pln)
    zt_pln = Zt.ravel()

    nx = 0
    ny = -1
    nz = 0
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)

    S_pln = np.empty_like(xt_pln)
    for idx, (xt, yt, zt) in enumerate(zip(xt_pln, yt_pln, zt_pln)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f*1e9, Is, dIsdx)
        S_pln[idx] = (Sx.real * nx + Sy.real * ny + Sz.real * nz) / n
    sPDn_pln = 1 / (2 * A) * elementwise_dblquad(points=np.c_[xt_pln, zt_pln],
                                                 values=S_pln,
                                                 degree=degree)
    
    # spherical surface
    N = 240 if f < 30 else 480
    r = 0.075
    theta = np.linspace(0, np.pi, N)
    phi = np.linspace(-np.pi, 0, N)
    Theta, Phi = np.meshgrid(theta, phi)

    x_sph, y_sph, z_sph = sph2cart(r, Theta.ravel(), Phi.ravel())
    xyz_sph = jnp.c_[x_sph, y_sph, z_sph]
    mask = np.where((xyz_sph[:, 0] >= xt_pln.min())
                    & (xyz_sph[:, 0] <= xt_pln.max())
                    & (xyz_sph[:, 2] >= zt_pln.min())
                    & (xyz_sph[:, 2] <= zt_pln.max()))[0]
    xt_sph = xyz_sph[mask, 0]
    yt_sph = xyz_sph[mask, 1] - xyz_sph[mask, 1].min()
    zt_sph = xyz_sph[mask, 2]

    nx, ny, nz = sph_normals(r, Theta.ravel()[mask], Phi.ravel()[mask])
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)

    A_sph = elementwise_dblquad(points=np.c_[xt_sph, zt_sph],
                                values=np.sin(Theta.ravel()[mask])*r**2/n,
                                degree=11) 

    S_sph = np.empty_like(xt_sph)
    for idx, (xt, yt, zt) in enumerate(zip(xt_sph, yt_sph, zt_sph)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f*1e9, Is, dIsdx)
        S_sph[idx] = (Sx.real * nx[idx] + Sy.real * ny[idx] + Sz.real * nz[idx])
    sPDn_sph = 1 / (2 * A_sph) * elementwise_dblquad(points=np.c_[xt_sph, zt_sph],
                                                     values=S_sph/n,
                                                     degree=degree)
    
    # cylndrical surface
    N = 220 if f < 30 else 440
    r = 0.0725
    theta = np.linspace(0, np.pi, N)
    Theta, Zt = np.meshgrid(-theta, _z)

    x_cyl, y_cyl, z_cyl = cyl2cart(r, Theta.ravel(), Zt.ravel())
    xyz_cyl = jnp.c_[x_cyl, y_cyl, z_cyl]
    mask = np.where((xyz_cyl[:, 0] >= xt_pln.min())
                     & (xyz_cyl[:, 0] <= xt_pln.max())
                     & (xyz_cyl[:, 2] >= zt_pln.min())
                     & (xyz_cyl[:, 2] <= zt_pln.max()))[0]
    xt_cyl = xyz_cyl[mask, 0]
    yt_cyl = xyz_cyl[mask, 1] - xyz_cyl[mask, 1].min()
    zt_cyl = xyz_cyl[mask, 2]

    nx, ny, nz = cyl_normals(r, Theta.ravel()[mask], zt_cyl)
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)

    A_cyl = elementwise_dblquad(points=np.c_[xt_cyl, zt_cyl],
                                values=1/n,
                                degree=11) 

    S_cyl = np.empty_like(xt_cyl)
    for idx, (xt, yt, zt) in enumerate(zip(xt_cyl, yt_cyl, zt_cyl)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f*1e9, Is, dIsdx)
        S_cyl[idx] = (Sx.real * nx[idx] + Sy.real * ny[idx] + Sz.real * nz[idx])
    sPDn_cyl = 1 / (2 * A_cyl) * elementwise_dblquad(points=np.c_[xt_cyl, zt_cyl],
                                                     values=S_cyl/n,
                                                     degree=degree)
    return sPDn_pln, sPDn_sph, sPDn_cyl


hs = np.array([2.5, 5, 10, 50, 150]) / -1000.  # separation distances
degree = 11  # integration degree

f = 6
sPDn_at6GHz = np.empty((hs.size, 4))
idx = 0
for h in tqdm(hs, desc=f'sPDn at {f}GHz'):
    sPDn_pln, sPDn_sph, sPDn_cyl = compute_sPDn(f, h, degree)
    sPDn_at6GHz[idx, :] = [h, sPDn_pln, sPDn_sph, sPDn_cyl]
    idx += 1
    
f = 26
sPDn_at26GHz = np.empty((hs.size, 4))
idx = 0
for h in tqdm(hs, desc=f'sPDn at {f}GHz'):
    sPDn_pln, sPDn_sph, sPDn_cyl = compute_sPDn(f, h, degree)
    sPDn_at26GHz[idx, :] = [h, sPDn_pln, sPDn_sph, sPDn_cyl]
    idx += 1
    
f = 60
sPDn_at60GHz = np.empty((hs.size, 4))
idx = 0
for h in tqdm(hs, desc=f'sPDn at {f}GHz'):
    sPDn_pln, sPDn_sph, sPDn_cyl =2 compute_sPDn(f, h, degree)
    sPDn_at60GHz[idx, :] = [h, sPDn_pln, sPDn_sph, sPDn_cyl]
    idx += 1
    
# save the data
np.save(os.path.join('data', f'sPDn_at6GHz'), sPDn_at6GHz)
np.save(os.path.join('data', f'sPDn_at26GHz'), sPDn_at26GHz)
np.save(os.path.join('data', f'sPDn_at60GHz'), sPDn_at60GHz)