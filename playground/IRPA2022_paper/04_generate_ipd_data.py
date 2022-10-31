import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from dosipy.field import poynting
from dosipy.utils.integrate import elementwise_dblquad
from utils import *


def compute_sPDn(f, h, r, degree, N=20):
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
    _x = jnp.linspace(-target_area[0]/2, target_area[0]/2, N)
    _z = jnp.linspace(-target_area[1]/2, target_area[1]/2, N)
    Xt, Zt = jnp.meshgrid(_x, _z)
    xt_pln = Xt.ravel()
    yt_pln = jnp.zeros_like(xt_pln)
    zt_pln = Zt.ravel()

    nx_pln = 0
    ny_pln = -1
    nz_pln = 0
    n_pln = np.sqrt(nx_pln ** 2 + ny_pln ** 2 + nz_pln ** 2)

    S_pln = np.empty_like(xt_pln)
    for idx, (xt, yt, zt) in enumerate(zip(xt_pln, yt_pln, zt_pln)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f*1e9, Is, dIsdx)
        S_pln[idx] = (Sx.real * nx_pln + Sy.real * ny_pln + Sz.real * nz_pln) / n_pln
    sPDn_pln = 1 / (2 * A) * elementwise_dblquad(points=np.c_[xt_pln, zt_pln],
                                                 values=S_pln,
                                                 degree=degree)
    
    # spherical surface
    alpha = 2 * np.arcsin(target_area[0]/2/r)
    theta = np.linspace(np.pi/2-alpha/2, np.pi/2+alpha/2, N)
    phi = np.linspace(np.pi-alpha/2, np.pi+alpha/2, N)
    Theta, Phi = np.meshgrid(theta, phi)
    yt_sph, xt_sph, zt_sph = sph2cart(r, Theta.ravel(), Phi.ravel())
    yt_sph -= yt_sph.min()

    ny_sph, nx_sph, nz_sph = sph_normals(r, Theta.ravel(), Phi.ravel())
    n_sph = np.sqrt(nx_sph ** 2 + ny_sph ** 2 + nz_sph ** 2)

    A_sph = elementwise_dblquad(points=np.c_[xt_sph, zt_sph],
                                values=np.sin(Theta.ravel())*r**2/n_sph,
                                degree=degree) 

    S_sph = np.empty_like(xt_sph)
    for idx, (xt, yt, zt) in enumerate(zip(xt_sph, yt_sph, zt_sph)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f*1e9, Is, dIsdx)
        S_sph[idx] = (Sx.real * nx_sph[idx] + Sy.real * ny_sph[idx] + Sz.real * nz_sph[idx])
    sPDn_sph = 1 / (2 * A_sph) * elementwise_dblquad(points=np.c_[xt_sph, zt_sph],
                                                     values=S_sph/n_sph,
                                                     degree=degree)
    
    # cylndrical surface
    alpha = 2 * np.arcsin(target_area[0]/2/r)
    theta = np.linspace(np.pi/2-alpha/2, np.pi/2+alpha/2, N)
    Theta, Zt = np.meshgrid(-theta, _z)
    xt_cyl, yt_cyl, zt_cyl = cyl2cart(r, Theta.ravel(), Zt.ravel())
    yt_cyl -= yt_cyl.min()

    nx_cyl, ny_cyl, nz_cyl = cyl_normals(r, Theta.ravel(), zt_cyl)
    n_cyl = np.sqrt(nx_cyl ** 2 + ny_cyl ** 2 + nz_cyl ** 2)

    A_cyl = elementwise_dblquad(points=np.c_[xt_cyl, zt_cyl],
                                values=1/n_cyl,
                                degree=degree) 

    S_cyl = np.empty_like(xt_cyl)
    for idx, (xt, yt, zt) in enumerate(zip(xt_cyl, yt_cyl, zt_cyl)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f*1e9, Is, dIsdx)
        S_cyl[idx] = (Sx.real * nx_cyl[idx] + Sy.real * ny_cyl[idx] + Sz.real * nz_cyl[idx])
    sPDn_cyl = 1 / (2 * A_cyl) * elementwise_dblquad(points=np.c_[xt_cyl, zt_cyl],
                                                     values=S_cyl/n_cyl,
                                                     degree=degree)
    return sPDn_pln, sPDn_sph, sPDn_cyl


hs = np.array([2, 5, 10, 50, 150]) / -1000.  # separation distances
rs = np.array([0.05, 0.07, 0.09, 0.12, 0.15])  # radii
degree = 11  # integration degree

f = 6
sPDn_at6GHz = np.empty((hs.size * rs.size, 5))
idx = 0
for r in tqdm(rs, desc=f'sPDn at {f}GHz'):
    for h in tqdm(hs):
        sPDn_pln, sPDn_sph, sPDn_cyl = compute_sPDn(f, h, r, degree)
        sPDn_at6GHz[idx, :] = [h, r, sPDn_pln, sPDn_sph, sPDn_cyl]
        idx += 1
    
f = 26
sPDn_at26GHz = np.empty((hs.size * rs.size, 5))
idx = 0
for r in tqdm(rs, desc=f'sPDn at {f}GHz'):
    for h in tqdm(hs):
        sPDn_pln, sPDn_sph, sPDn_cyl = compute_sPDn(f, h, r, degree)
        sPDn_at26GHz[idx, :] = [h, r, sPDn_pln, sPDn_sph, sPDn_cyl]
        idx += 1
    
f = 60
sPDn_at60GHz = np.empty((hs.size * rs.size, 5))
idx = 0
for r in tqdm(rs, desc=f'sPDn at {f}GHz'):
    for h in tqdm(hs):
        sPDn_pln, sPDn_sph, sPDn_cyl = compute_sPDn(f, h, r, degree)
        sPDn_at60GHz[idx, :] = [h, r, sPDn_pln, sPDn_sph, sPDn_cyl]
        idx += 1
    
f = 90
sPDn_at90GHz = np.empty((hs.size * rs.size, 5))
idx = 0
for r in tqdm(rs, desc=f'sPDn at {f}GHz'):
    for h in tqdm(hs):
        sPDn_pln, sPDn_sph, sPDn_cyl = compute_sPDn(f, h, r, degree)
        sPDn_at90GHz[idx, :] = [h, r, sPDn_pln, sPDn_sph, sPDn_cyl]
        idx += 1
    
# save the data
np.save(os.path.join('data', f'sPDn_at6GHz'), sPDn_at6GHz)
np.save(os.path.join('data', f'sPDn_at26GHz'), sPDn_at26GHz)
np.save(os.path.join('data', f'sPDn_at60GHz'), sPDn_at60GHz)
np.save(os.path.join('data', f'sPDn_at90GHz'), sPDn_at90GHz)
