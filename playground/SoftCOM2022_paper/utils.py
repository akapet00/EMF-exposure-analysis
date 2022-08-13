import numpy as np

from dosipy.field import poynting
from dosipy.utils.derive import holoborodko
from dosipy.utils.integrate import elementwise_dblquad
from dosipy.utils.dataloader import load_antenna_el_properties


def cart2sph(x, y, z):
    """Return spherical given Cartesain coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph2cart(r, theta, phi):
    """Return Cartesian given Spherical coordinates."""
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def sph_normals(r, theta, phi):
    """Return unit vector field components normal to spherical
    surface."""
    nx = r ** 2 * np.cos(phi) * np.sin(theta) ** 2 
    ny = r ** 2 * np.sin(phi) * np.sin(theta) ** 2
    nz = r ** 2 * np.cos(theta) * np.sin(theta)
    return nx, ny, nz


def IPD(r, f, d, deg, edge_length):
    """Return incident power density over the planar and spherical
    surfaces."""
    # source
    data = load_antenna_el_properties(f)
    xs = data.x.to_numpy()
    xs -= xs.max() / 2
    ys = np.zeros_like(xs) + d
    zs = np.zeros_like(xs)
    Is = np.abs(data.ireal.to_numpy() + 1j * data.iimag.to_numpy())
    dx = xs[1] - xs[0]
    Is_x = holoborodko(Is, dx)
    
    # planar target
    N = 33
    x = np.linspace(-edge_length/2, edge_length/2, N)
    z = np.linspace(-edge_length/2, edge_length/2, N)
    X, Z = np.meshgrid(x, z)
    xt_pln = X.ravel()
    yt_pln = np.zeros_like(xt_pln)
    zt_pln = Z.ravel()
    
    nx_pln = 0
    ny_pln = -1
    nz_pln = 0
    n_len_pln = np.sqrt(nx_pln ** 2 + ny_pln ** 2 + nz_pln ** 2)
    
    A_pln = edge_length ** 2
    
    S_pln = np.empty_like(xt_pln)
    for idx, (xt, yt, zt) in enumerate(zip(xt_pln, yt_pln, zt_pln)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f, Is, Is_x)
        S_pln[idx] = Sx.real * nx_pln + Sy.real * ny_pln + Sz.real * nz_pln
    sPDn_pln = 1 / (2 * A_pln) * elementwise_dblquad(points=np.c_[xt_pln, zt_pln],
                                                     values=S_pln/n_len_pln,
                                                     degree=deg)
    
    # spherical target
    alpha = 2 * np.arcsin(edge_length/2/r)
    N = 33
    theta = np.linspace(np.pi/2 - alpha/2, np.pi/2 + alpha/2, N)
    phi = np.linspace(np.pi-alpha/2, np.pi+alpha/2, N)
    Theta, Phi = np.meshgrid(theta, phi)
    yt_sph, xt_sph, zt_sph = sph2cart(r, Theta.ravel(), Phi.ravel())
    yt_sph -= yt_sph.min()
    
    ny_sph, nx_sph, nz_sph = sph_normals(r, Theta.ravel(), Phi.ravel())
    n_len_sph = np.sqrt(nx_sph ** 2 + ny_sph ** 2 + nz_sph ** 2)
    
    A_sph = elementwise_dblquad(points=np.c_[xt_sph, zt_sph],
                                values=np.sin(Theta.ravel())*r**2/n_len_sph,
                                degree=deg)
    
    S_sph = np.empty_like(xt_sph)
    for idx, (xt, yt, zt) in enumerate(zip(xt_sph, yt_sph, zt_sph)):
        Sx, Sy, Sz = poynting(xt, yt, zt, xs, ys, zs, f, Is, Is_x)
        S_sph[idx] = Sx.real * nx_sph[idx] + Sy.real * ny_sph[idx] + Sz.real * nz_sph[idx]
    sPDn_sph = 1 / (2 * A_sph) * elementwise_dblquad(points=np.c_[xt_sph, zt_sph],
                                                     values=S_sph/n_len_sph,
                                                     degree=deg)
    return sPDn_pln, sPDn_sph
